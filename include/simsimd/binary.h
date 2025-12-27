/**
 *  @brief SIMD-accelerated Binary Similarity Measures.
 *  @file include/simsimd/binary.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *
 *  Contains following similarity measures:
 *
 *  - Bit-level Hamming distance → `u32` counter
 *  - Bit-level Jaccard distance (Tanimoto coefficient) → `f32` ratio
 *  - Jaccard distance for `u32` integral MinHash vectors from StringZilla → `f32` ratio
 *  - TODO: Weighted Jaccard distance for `u32` integral Count-Min-Sketch vectors → `f32` ratio
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
 *  On binary vectors, when computing Jaccard distance, the CPU often struggles to compute the
 *  large number of required population counts. There are several instructions we should keep in mind:
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
 *  // 1024-dimensional binary vectors, one query and four targets
 *  simsimd_b8_t query[128], target_first[128], target_second[128], target_third[128], target_fourth[128];
 *  // Precomputed popcount of 'a' as f32
 *  simsimd_f32_t query_popcount = ...;
 *  simsimd_f32_t target_popcount_first = ..., target_popcount_second = ...;
 *
 *  simsimd_jaccard_b512_state_ice_t state_first, state_second, state_third, state_fourth;
 *  simsimd_jaccard_b512_init_ice(&state_first);
 *  simsimd_jaccard_b512_init_ice(&state_second);
 *  simsimd_jaccard_b512_init_ice(&state_third);
 *  simsimd_jaccard_b512_init_ice(&state_fourth);
 *  simsimd_jaccard_b512_update_ice(&state_first, &query[0], &target_first[0]); // First 512 bits
 *  simsimd_jaccard_b512_update_ice(&state_first, &query[64], &target_first[64]); // Second 512 bits
 *  // ... update state_second, state_third, state_fourth similarly ...
 *
 *  simsimd_f32_t results[4];
 *  simsimd_jaccard_b512_finalize_ice(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, target_popcount_first, target_popcount_second,
 *      target_popcount_third, target_popcount_fourth, results);
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
 *  @section Finalize Output Types
 *
 *  Jaccard similarity finalize outputs to f32:
 *  - Jaccard = intersection / union, always in [0.0, 1.0]
 *  - f32 provides ~7 decimal digits, far exceeding practical needs
 *  - Matches spatial.h convention for non-f64 distance outputs
 *  - Reduces memory footprint in large-scale binary similarity search
 *
 *  The intersection and union counts are u64 internally for correctness,
 *  but the final ratio fits comfortably in f32.
 *
 */
#ifndef SIMSIMD_BINARY_H
#define SIMSIMD_BINARY_H

#include "types.h"

#include "reduce.h"

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
                                        simsimd_u32_t *result);

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
                                        simsimd_f32_t *result);

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
                                         simsimd_f32_t *result);

/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_serial(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                              simsimd_u32_t *result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_serial(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                              simsimd_f32_t *result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_serial(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *result);

typedef struct simsimd_jaccard_b128_state_serial_t simsimd_jaccard_b128_state_serial_t;
/** @copydoc simsimd_jaccard_b128_state_serial_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_init_serial(simsimd_jaccard_b128_state_serial_t *state);
/** @copydoc simsimd_jaccard_b128_state_serial_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_update_serial(simsimd_jaccard_b128_state_serial_t *state,
                                                         simsimd_b128_vec_t a, simsimd_b128_vec_t b);
/** @copydoc simsimd_jaccard_b128_state_serial_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_finalize_serial(
    simsimd_jaccard_b128_state_serial_t const *state_a, simsimd_jaccard_b128_state_serial_t const *state_b,
    simsimd_jaccard_b128_state_serial_t const *state_c, simsimd_jaccard_b128_state_serial_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_u32_t *result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);

typedef struct simsimd_jaccard_b128_state_neon_t simsimd_jaccard_b128_state_neon_t;
/** @copydoc simsimd_jaccard_b128_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_init_neon(simsimd_jaccard_b128_state_neon_t *state);
/** @copydoc simsimd_jaccard_b128_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_update_neon(simsimd_jaccard_b128_state_neon_t *state, uint8x16_t a,
                                                       uint8x16_t b);
/** @copydoc simsimd_jaccard_b128_state_neon_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b128_finalize_neon(
    simsimd_jaccard_b128_state_neon_t const *state_a, simsimd_jaccard_b128_state_neon_t const *state_b,
    simsimd_jaccard_b128_state_neon_t const *state_c, simsimd_jaccard_b128_state_neon_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_u32_t *result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_sve(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_u32_t *result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_f32_t *result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *result);

typedef struct simsimd_jaccard_b256_state_haswell_t simsimd_jaccard_b256_state_haswell_t;
/** @copydoc simsimd_jaccard_b256_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b256_init_haswell(simsimd_jaccard_b256_state_haswell_t *state);
/** @copydoc simsimd_jaccard_b256_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b256_update_haswell(simsimd_jaccard_b256_state_haswell_t *state,
                                                          simsimd_b256_vec_t a, simsimd_b256_vec_t b);
/** @copydoc simsimd_jaccard_b256_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b256_finalize_haswell(
    simsimd_jaccard_b256_state_haswell_t const *state_a, simsimd_jaccard_b256_state_haswell_t const *state_b,
    simsimd_jaccard_b256_state_haswell_t const *state_c, simsimd_jaccard_b256_state_haswell_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_ICE
/** @copydoc simsimd_hamming_b8 */
SIMSIMD_PUBLIC void simsimd_hamming_b8_ice(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_u32_t *result);
/** @copydoc simsimd_jaccard_b8 */
SIMSIMD_PUBLIC void simsimd_jaccard_b8_ice(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_jaccard_u32 */
SIMSIMD_PUBLIC void simsimd_jaccard_u32_ice(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);

typedef struct simsimd_jaccard_b512_state_ice_t simsimd_jaccard_b512_state_ice_t;
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_ice(simsimd_jaccard_b512_state_ice_t *state);
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_ice(simsimd_jaccard_b512_state_ice_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b);
/** @copydoc simsimd_jaccard_b512_state_ice_t */
SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_ice(
    simsimd_jaccard_b512_state_ice_t const *state_a, simsimd_jaccard_b512_state_ice_t const *state_b,
    simsimd_jaccard_b512_state_ice_t const *state_c, simsimd_jaccard_b512_state_ice_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_ICE

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
                                              simsimd_u32_t *result) {
    simsimd_u32_t differences = 0;
    for (simsimd_size_t i = 0; i != n_words; ++i) differences += simsimd_popcount_b8(a[i] ^ b[i]);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_serial(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                              simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0, union_count = 0;
    for (simsimd_size_t i = 0; i != n_words; ++i)
        intersection_count += simsimd_popcount_b8(a[i] & b[i]), union_count += simsimd_popcount_b8(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)union_count : 1.0f;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_serial(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0;
    for (simsimd_size_t i = 0; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)n : 1.0f;
}

/**
 *  @brief Running state for 128-bit Jaccard accumulation (serial/portable).
 *
 *  Portable implementation using scalar popcount. The update receives 128-bit
 *  chunks as `simsimd_b128_vec_t`. State uses u64 accumulator for large vectors.
 */
typedef struct simsimd_jaccard_b128_state_serial_t {
    simsimd_u64_t intersection_count;
} simsimd_jaccard_b128_state_serial_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b128_init_serial(simsimd_jaccard_b128_state_serial_t *state) {
    state->intersection_count = 0;
}

SIMSIMD_INTERNAL void simsimd_jaccard_b128_update_serial(simsimd_jaccard_b128_state_serial_t *state,
                                                         simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_u64_t intersection_low = a.u64s[0] & b.u64s[0];
    simsimd_u64_t intersection_high = a.u64s[1] & b.u64s[1];
    state->intersection_count += _simsimd_u64_popcount(intersection_low);
    state->intersection_count += _simsimd_u64_popcount(intersection_high);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b128_finalize_serial(
    simsimd_jaccard_b128_state_serial_t const *state_a, simsimd_jaccard_b128_state_serial_t const *state_b,
    simsimd_jaccard_b128_state_serial_t const *state_c, simsimd_jaccard_b128_state_serial_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results) {

    simsimd_f32_t intersection_a = (simsimd_f32_t)state_a->intersection_count;
    simsimd_f32_t intersection_b = (simsimd_f32_t)state_b->intersection_count;
    simsimd_f32_t intersection_c = (simsimd_f32_t)state_c->intersection_count;
    simsimd_f32_t intersection_d = (simsimd_f32_t)state_d->intersection_count;

    simsimd_f32_t union_a = query_popcount + target_popcount_a - intersection_a;
    simsimd_f32_t union_b = query_popcount + target_popcount_b - intersection_b;
    simsimd_f32_t union_c = query_popcount + target_popcount_c - intersection_c;
    simsimd_f32_t union_d = query_popcount + target_popcount_d - intersection_d;

    results[0] = (union_a != 0) ? 1.0f - intersection_a / union_a : 1.0f;
    results[1] = (union_b != 0) ? 1.0f - intersection_b / union_b : 1.0f;
    results[2] = (union_c != 0) ? 1.0f - intersection_c / union_c : 1.0f;
    results[3] = (union_d != 0) ? 1.0f - intersection_d / union_d : 1.0f;
}

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_u32_t *result) {
    simsimd_u32_t differences = 0;
    simsimd_size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_words) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            uint8x16_t xor_popcount_u8x16 = vcntq_u8(veorq_u8(a_u8x16, b_u8x16));
            popcount_u8x16 = vaddq_u8(popcount_u8x16, xor_popcount_u8x16);
        }
        differences += _simsimd_reduce_add_u8x16_neon(popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_words; ++i) differences += simsimd_popcount_b8(a[i] ^ b[i]);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_neon(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                            simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0, union_count = 0;
    simsimd_size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_words) {
        uint8x16_t intersection_popcount_u8x16 = vdupq_n_u8(0);
        uint8x16_t union_popcount_u8x16 = vdupq_n_u8(0);
        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            intersection_popcount_u8x16 = vaddq_u8(intersection_popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
            union_popcount_u8x16 = vaddq_u8(union_popcount_u8x16, vcntq_u8(vorrq_u8(a_u8x16, b_u8x16)));
        }
        intersection_count += _simsimd_reduce_add_u8x16_neon(intersection_popcount_u8x16);
        union_count += _simsimd_reduce_add_u8x16_neon(union_popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_words; ++i)
        intersection_count += simsimd_popcount_b8(a[i] & b[i]), union_count += simsimd_popcount_b8(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)union_count : 1.0f;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0;
    simsimd_size_t i = 0;
    uint32x4_t intersection_count_u32x4 = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i);
        uint32x4_t b_u32x4 = vld1q_u32(b + i);
        uint32x4_t equality_mask = vceqq_u32(a_u32x4, b_u32x4);
        intersection_count_u32x4 = vaddq_u32(intersection_count_u32x4, vshrq_n_u32(equality_mask, 31));
    }
    intersection_count += vaddvq_u32(intersection_count_u32x4);
    for (; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)n : 1.0f;
}

/**
 *  @brief Running state for 128-bit Jaccard accumulation on NEON.
 *
 *  This is a minimal state variant designed for processing one 128-bit chunk
 *  at a time, matching the natural ARM NEON register size. Use this when:
 *  - Processing streams where 128-bit granularity is natural
 *  - Memory bandwidth is the bottleneck (minimal state overhead)
 *  - Integration with systems that already work in 128-bit chunks
 *
 *  The state uses `uint32x4_t` vector accumulation to defer horizontal reduction
 *  to finalize. Each update uses `vaddq_u32` (1 cycle) instead of `vaddvq_u32`
 *  (2-3 cycles), improving throughput for large vectors.
 *
 *  @code{.c}
 *  // 256-bit binary vectors (2 x 128-bit chunks), one query and four targets
 *  simsimd_b8_t query[32], target_a[32], target_b[32], target_c[32], target_d[32];
 *  simsimd_f32_t query_popcount = 100.0f;
 *  simsimd_f32_t popcount_a = 95.0f, popcount_b = 110.0f, popcount_c = 88.0f, popcount_d = 102.0f;
 *
 *  simsimd_jaccard_b128_state_neon_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b128_init_neon(&state_a);
 *  simsimd_jaccard_b128_init_neon(&state_b);
 *  simsimd_jaccard_b128_init_neon(&state_c);
 *  simsimd_jaccard_b128_init_neon(&state_d);
 *
 *  // Update for first 128-bit chunk
 *  uint8x16_t query_chunk0 = vld1q_u8(query);
 *  simsimd_jaccard_b128_update_neon(&state_a, query_chunk0, vld1q_u8(target_a));
 *  simsimd_jaccard_b128_update_neon(&state_b, query_chunk0, vld1q_u8(target_b));
 *  simsimd_jaccard_b128_update_neon(&state_c, query_chunk0, vld1q_u8(target_c));
 *  simsimd_jaccard_b128_update_neon(&state_d, query_chunk0, vld1q_u8(target_d));
 *
 *  // Update for second 128-bit chunk
 *  uint8x16_t query_chunk1 = vld1q_u8(query + 16);
 *  simsimd_jaccard_b128_update_neon(&state_a, query_chunk1, vld1q_u8(target_a + 16));
 *  simsimd_jaccard_b128_update_neon(&state_b, query_chunk1, vld1q_u8(target_b + 16));
 *  simsimd_jaccard_b128_update_neon(&state_c, query_chunk1, vld1q_u8(target_c + 16));
 *  simsimd_jaccard_b128_update_neon(&state_d, query_chunk1, vld1q_u8(target_d + 16));
 *
 *  // Finalize all 4 states at once
 *  simsimd_f32_t results[4];
 *  simsimd_jaccard_b128_finalize_neon(&state_a, &state_b, &state_c, &state_d,
 *      query_popcount, popcount_a, popcount_b, popcount_c, popcount_d, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b128_state_neon_t {
    uint32x4_t intersection_count_u32x4;
} simsimd_jaccard_b128_state_neon_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b128_init_neon(simsimd_jaccard_b128_state_neon_t *state) {
    state->intersection_count_u32x4 = vdupq_n_u32(0);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b128_update_neon(simsimd_jaccard_b128_state_neon_t *state, uint8x16_t a,
                                                       uint8x16_t b) {

    // Process one 128-bit chunk (native ARM NEON register size).
    // Uses vector accumulation - horizontal sum deferred to finalize.
    //
    // ARM NEON instruction characteristics:
    //   `vandq_u8`:   Bitwise AND, 1 cycle latency
    //   `vcntq_u8`:   Byte popcount (16 bytes -> 16 popcounts), 1-2 cycles
    //   `vpaddlq_u8`: Pairwise widening add u8->u16, 1 cycle
    //   `vpaddlq_u16`: Pairwise widening add u16->u32, 1 cycle
    //   `vaddq_u32`:  Vector add u32x4, 1 cycle
    // Total: ~5-6 cycles per 128-bit chunk (no horizontal sum penalty per update)

    // Step 1: Compute intersection bits (A AND B)
    uint8x16_t intersection_u8x16 = vandq_u8(a, b);

    // Step 2: Byte-level popcount - each byte contains count of set bits (0-8)
    uint8x16_t popcount_u8x16 = vcntq_u8(intersection_u8x16);

    // Step 3: Pairwise widening reduction chain
    // u8x16 -> u16x8: pairs of adjacent bytes summed into 16-bit
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    // u16x8 -> u32x4: pairs of 16-bit values summed into 32-bit
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);

    // Step 4: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vaddq_u32(state->intersection_count_u32x4, popcount_u32x4);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b128_finalize_neon(
    simsimd_jaccard_b128_state_neon_t const *state_a, simsimd_jaccard_b128_state_neon_t const *state_b,
    simsimd_jaccard_b128_state_neon_t const *state_c, simsimd_jaccard_b128_state_neon_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results) {

    // Horizontal sum each state's vector accumulator via `vaddvq_u32` (ARMv8.1+, 2-3 cycles)
    // This is done once at finalize, not per-update, for better throughput.
    uint32x4_t intersection_u32x4 = (uint32x4_t) {
        vaddvq_u32(state_a->intersection_count), vaddvq_u32(state_b->intersection_count),
        vaddvq_u32(state_c->intersection_count), vaddvq_u32(state_d->intersection_count)};
    float32x4_t intersection_f32x4 = vcvtq_f32_u32(intersection_u32x4);

    // Compute union using |A OR B| = |A| + |B| - |A AND B|
    float32x4_t query_f32x4 = vdupq_n_f32(query_popcount);
    float32x4_t targets_f32x4 = (float32x4_t) {target_popcount_a, target_popcount_b, target_popcount_c,
                                               target_popcount_d};
    float32x4_t union_f32x4 = vsubq_f32(vaddq_f32(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case (empty vectors -> distance = 1.0)
    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    uint32x4_t zero_union_mask = vceqq_f32(union_f32x4, vdupq_n_f32(0.0f));
    float32x4_t safe_union_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, union_f32x4);

    // Fast reciprocal with Newton-Raphson refinement:
    //   `vrecpeq_f32`: ~12-bit estimate, 1 cycle
    //   `vrecpsq_f32`: Newton-Raphson step computes (2 - a*b), 1 cycle
    //   `vmulq_f32`: multiply, 1 cycle
    // One N-R iteration: ~24-bit accuracy, sufficient for f32 (23 mantissa bits).
    // Total: ~3-4 cycles vs ~10-14 cycles for division.
    float32x4_t union_reciprocal_f32x4 = vrecpeq_f32(safe_union_f32x4);
    union_reciprocal_f32x4 = vmulq_f32(union_reciprocal_f32x4, vrecpsq_f32(safe_union_f32x4, union_reciprocal_f32x4));

    // Compute Jaccard distance = 1 - intersection/union
    float32x4_t ratio_f32x4 = vmulq_f32(intersection_f32x4, union_reciprocal_f32x4);
    float32x4_t jaccard_f32x4 = vsubq_f32(one_f32x4, ratio_f32x4);
    float32x4_t result_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, jaccard_f32x4);

    vst1q_f32(results, result_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_u32_t *result) {

    // On very small register sizes, NEON is at least as fast as SVE.
    simsimd_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        simsimd_hamming_b8_neon(a, b, n_words, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    simsimd_size_t i = 0, cycle = 0;
    simsimd_u32_t differences = 0;
    svuint8_t popcount_u8 = svdup_n_u8(0);
    svbool_t const all_predicate = svptrue_b8();
    while (i < n_words) {
        do {
            svbool_t active_predicate = svwhilelt_b8((unsigned int)i, (unsigned int)n_words);
            svuint8_t a_u8 = svld1_u8(active_predicate, a + i);
            svuint8_t b_u8 = svld1_u8(active_predicate, b + i);
            popcount_u8 = svadd_u8_z(all_predicate, popcount_u8,
                                     svcnt_u8_x(all_predicate, sveor_u8_m(all_predicate, a_u8, b_u8)));
            i += words_per_register;
            ++cycle;
        } while (i < n_words && cycle < 31);
        differences += svaddv_u8(all_predicate, popcount_u8);
        popcount_u8 = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_sve(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_f32_t *result) {

    // On very small register sizes, NEON is at least as fast as SVE.
    simsimd_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        simsimd_jaccard_b8_neon(a, b, n_words, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    simsimd_size_t i = 0, cycle = 0;
    simsimd_u32_t intersection_count = 0, union_count = 0;
    svuint8_t intersection_popcount_u8 = svdup_n_u8(0);
    svuint8_t union_popcount_u8 = svdup_n_u8(0);
    svbool_t const all_predicate = svptrue_b8();
    while (i < n_words) {
        do {
            svbool_t active_predicate = svwhilelt_b8((unsigned int)i, (unsigned int)n_words);
            svuint8_t a_u8 = svld1_u8(active_predicate, a + i);
            svuint8_t b_u8 = svld1_u8(active_predicate, b + i);
            intersection_popcount_u8 = svadd_u8_z(all_predicate, intersection_popcount_u8,
                                                  svcnt_u8_x(all_predicate, svand_u8_m(all_predicate, a_u8, b_u8)));
            union_popcount_u8 = svadd_u8_z(all_predicate, union_popcount_u8,
                                           svcnt_u8_x(all_predicate, svorr_u8_m(all_predicate, a_u8, b_u8)));
            i += words_per_register;
            ++cycle;
        } while (i < n_words && cycle < 31);
        intersection_count += svaddv_u8(all_predicate, intersection_popcount_u8);
        intersection_popcount_u8 = svdup_n_u8(0);
        union_count += svaddv_u8(all_predicate, union_popcount_u8);
        union_popcount_u8 = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = (union_count != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)union_count : 1.0f;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_sve(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_size_t const words_per_register = svcntw();
    simsimd_size_t i = 0;
    simsimd_u32_t intersection_count = 0;
    while (i < n) {
        svbool_t active_predicate = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svuint32_t a_u32 = svld1_u32(active_predicate, a + i);
        svuint32_t b_u32 = svld1_u32(active_predicate, b + i);
        svbool_t equality_predicate = svcmpeq_u32(active_predicate, a_u32, b_u32);
        intersection_count += svcntp_b32(active_predicate, equality_predicate);
        i += words_per_register;
    }
    *result = (n != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)n : 1.0f;
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
                                           simsimd_u32_t *result) {

    simsimd_u32_t xor_count;
    // It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_words <= 64) { // Up to 512 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        __m512i xor_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_u8x64, b_u8x64));
        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    else if (n_words <= 128) { // Up to 1024 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 64);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8));
    }
    else if (n_words <= 192) { // Up to 1536 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 128);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(
            xor_popcount_three_u64x8, _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else if (n_words <= 256) { // Up to 2048 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 192);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        __m512i xor_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_four_u8x64, b_four_u8x64));
        xor_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(xor_popcount_four_u64x8, xor_popcount_three_u64x8),
                             _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else {
        __m512i xor_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    simsimd_hamming_b8_ice_cycle:
        if (n_words < 64) {
            __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
            a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
            n_words = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_words -= 64;
        }
        __m512i xor_u8x64 = _mm512_xor_si512(a_u8x64, b_u8x64);
        xor_popcount_u64x8 = _mm512_add_epi64(xor_popcount_u64x8, _mm512_popcnt_epi64(xor_u8x64));
        if (n_words) goto simsimd_hamming_b8_ice_cycle;

        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    *result = xor_count;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_ice(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                           simsimd_f32_t *result) {

    simsimd_u32_t intersection_count = 0, union_count = 0;
    //  It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_words <= 64) { // Up to 512 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b);
        __m512i intersection_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_u8x64, b_u8x64));
        __m512i union_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_u8x64, b_u8x64));
        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    else if (n_words <= 128) { // Up to 1024 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 64);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8));
        union_count = _mm512_reduce_add_epi64(_mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8));
    }
    else if (n_words <= 192) { // Up to 1536 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 128);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        intersection_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(intersection_popcount_three_u64x8,
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(union_popcount_three_u64x8,
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else if (n_words <= 256) { // Up to 2048 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 192);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        __m512i intersection_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_four_u8x64, b_four_u8x64));
        __m512i union_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_four_u8x64, b_four_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(intersection_popcount_four_u64x8, intersection_popcount_three_u64x8),
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(union_popcount_four_u64x8, union_popcount_three_u64x8),
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else {
        __m512i intersection_popcount_u64x8 = _mm512_setzero_si512();
        __m512i union_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    simsimd_jaccard_b8_ice_cycle:
        if (n_words < 64) {
            __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
            a_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b);
            n_words = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_words -= 64;
        }
        __m512i intersection_u8x64 = _mm512_and_si512(a_u8x64, b_u8x64);
        __m512i union_u8x64 = _mm512_or_si512(a_u8x64, b_u8x64);
        intersection_popcount_u64x8 = _mm512_add_epi64(intersection_popcount_u64x8,
                                                       _mm512_popcnt_epi64(intersection_u8x64));
        union_popcount_u64x8 = _mm512_add_epi64(union_popcount_u64x8, _mm512_popcnt_epi64(union_u8x64));
        if (n_words) goto simsimd_jaccard_b8_ice_cycle;

        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    *result = (union_count != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)union_count : 1.0f;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_ice(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0;
    simsimd_size_t n_remaining = n;
    for (; n_remaining >= 16; n_remaining -= 16, a += 16, b += 16) {
        __m512i a_u32x16 = _mm512_loadu_epi32(a);
        __m512i b_u32x16 = _mm512_loadu_epi32(b);
        __mmask16 equality_mask = _mm512_cmpeq_epi32_mask(a_u32x16, b_u32x16);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    if (n_remaining) {
        __mmask16 load_mask = (__mmask16)_bzhi_u32(0xFFFF, n_remaining);
        __m512i a_u32x16 = _mm512_maskz_loadu_epi32(load_mask, a);
        __m512i b_u32x16 = _mm512_maskz_loadu_epi32(load_mask, b);
        __mmask16 equality_mask = _mm512_cmpeq_epi32_mask(a_u32x16, b_u32x16) & load_mask;
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    *result = (n != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)n : 1.0f;
}

/**
 *  @brief Running state for 512-bit Jaccard accumulation on Ice Lake.
 *  @code{.c}
 *  // 1024-dimensional binary vectors, one query and four targets
 *  simsimd_b8_t query[128], target_first[128], target_second[128], target_third[128], target_fourth[128];
 *  // Precomputed popcount of 'a' as f32
 *  simsimd_f32_t query_popcount = ...;
 *  simsimd_f32_t target_popcount_first = ..., target_popcount_second = ...;
 *
 *  simsimd_jaccard_b512_state_ice_t state_first, state_second, state_third, state_fourth;
 *  simsimd_jaccard_b512_init_ice(&state_first);
 *  simsimd_jaccard_b512_init_ice(&state_second);
 *  simsimd_jaccard_b512_init_ice(&state_third);
 *  simsimd_jaccard_b512_init_ice(&state_fourth);
 *  simsimd_jaccard_b512_update_ice(&state_first, &query[0], &target_first[0]); // First 512 bits
 *  simsimd_jaccard_b512_update_ice(&state_first, &query[64], &target_first[64]); // Second 512 bits
 *  // ... update state_second, state_third, state_fourth similarly ...
 *
 *  simsimd_f32_t results[4];
 *  simsimd_jaccard_b512_finalize_ice(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, target_popcount_first, target_popcount_second,
 *      target_popcount_third, target_popcount_fourth, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b512_state_ice_t {
    __m512i intersection_count_i64x8;
} simsimd_jaccard_b512_state_ice_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b512_init_ice(simsimd_jaccard_b512_state_ice_t *state) {
    state->intersection_count_i64x8 = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_update_ice(simsimd_jaccard_b512_state_ice_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    state->intersection_count_i64x8 = _mm512_add_epi64(state->intersection_count_i64x8,
                                                       _mm512_popcnt_epi64(_mm512_and_si512(a.zmm, b.zmm)));
}

SIMSIMD_INTERNAL void simsimd_jaccard_b512_finalize_ice(
    simsimd_jaccard_b512_state_ice_t const *state_a, simsimd_jaccard_b512_state_ice_t const *state_b,
    simsimd_jaccard_b512_state_ice_t const *state_c, simsimd_jaccard_b512_state_ice_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results) {

    // Port-optimized 4-way horizontal reduction using early i64→i32 truncation.
    //
    // Key insight: `_mm_hadd_epi32` uses ports p01, not p5, avoiding the shuffle bottleneck.
    // By truncating to i32 early, we can use hadd for reduction instead of expensive shuffles.
    //
    // Ice Lake execution ports:
    //   p0:   Division, reciprocal (`VRCP14PS`: 4cy latency, 1/cy throughput)
    //   p01:  FP mul/add/fma, hadd (`VMULPS`/`VPHADDD`: 3cy latency, 0.5/cy throughput)
    //   p015: Integer add (`VPADDD`: 1cy latency, 0.33/cy throughput)
    //   p5:   Shuffles/extracts (`VEXTRACTI128`: 3cy latency, 1/cy throughput)

    // Step 1: Truncate 8×i64 → 8×i32 per state (fits in YMM)
    // `VPMOVQD` (ZMM→YMM): 4cy latency, 0.5/cy throughput, port p01
    __m256i a_i32x8 = _mm512_cvtepi64_epi32(state_a->intersection_count_i64x8);
    __m256i b_i32x8 = _mm512_cvtepi64_epi32(state_b->intersection_count_i64x8);
    __m256i c_i32x8 = _mm512_cvtepi64_epi32(state_c->intersection_count_i64x8);
    __m256i d_i32x8 = _mm512_cvtepi64_epi32(state_d->intersection_count_i64x8);

    // Step 2: Reduce 8×i32 → 4×i32 (add high 128-bit lane to low)
    // `VEXTRACTI128`: 3cy latency, 1/cy throughput, port p5
    // `VPADDD` (XMM): 1cy latency, 0.33/cy throughput, ports p015
    __m128i a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(a_i32x8), _mm256_extracti128_si256(a_i32x8, 1));
    __m128i b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(b_i32x8), _mm256_extracti128_si256(b_i32x8, 1));
    __m128i c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(c_i32x8), _mm256_extracti128_si256(c_i32x8, 1));
    __m128i d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(d_i32x8), _mm256_extracti128_si256(d_i32x8, 1));

    // Step 3: Reduce 4×i32 → 2×i32 using horizontal add (uses p01, not p5!)
    // `VPHADDD` (XMM): 3cy latency, 0.5/cy throughput, ports p01
    __m128i ab_i32x4 = _mm_hadd_epi32(a_i32x4, b_i32x4); // [a01, a23, b01, b23]
    __m128i cd_i32x4 = _mm_hadd_epi32(c_i32x4, d_i32x4); // [c01, c23, d01, d23]

    // Step 4: Reduce 2×i32 → 1×i32 per state (final horizontal add)
    __m128i intersection_i32x4 = _mm_hadd_epi32(ab_i32x4, cd_i32x4); // [a, b, c, d]

    // Step 5: Direct i32 → f32 conversion (simpler than i64→f64→f32 path)
    // `VCVTDQ2PS` (XMM): 4cy latency, 0.5/cy throughput, port p01
    __m128 intersection_f32x4 = _mm_cvtepi32_ps(intersection_i32x4);

    // Compute Jaccard distance: 1 - intersection / union
    // where union = query_popcount + target_popcount - intersection
    __m128 query_f32x4 = _mm_set1_ps(query_popcount);
    __m128 targets_f32x4 = _mm_setr_ps(target_popcount_a, target_popcount_b, target_popcount_c, target_popcount_d);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case: if union == 0, result = 1.0
    __m128 zero_union_mask = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_mask);

    // Fast reciprocal with Newton-Raphson refinement:
    //   `VRCP14PS`: 4cy latency, 1/cy throughput, port p0 (~14-bit precision)
    //   Newton-Raphson: rcp' = rcp * (2 - x * rcp) doubles precision to ~28 bits
    //   `VFNMADD`: 4cy latency, 0.5/cy throughput, ports p01
    //   `VMULPS`: 4cy latency, 0.5/cy throughput, ports p01
    // Total: ~12cy vs `VDIVPS` 11cy latency but 3cy throughput - NR wins on throughput
    __m128 union_reciprocal_f32x4 = _mm_rcp14_ps(safe_union_f32x4);
    union_reciprocal_f32x4 = _mm_mul_ps(union_reciprocal_f32x4,
                                        _mm_fnmadd_ps(safe_union_f32x4, union_reciprocal_f32x4, _mm_set1_ps(2.0f)));

    __m128 ratio_f32x4 = _mm_mul_ps(intersection_f32x4, union_reciprocal_f32x4);
    __m128 jaccard_f32x4 = _mm_sub_ps(one_f32x4, ratio_f32x4);
    __m128 result_f32x4 = _mm_blendv_ps(jaccard_f32x4, one_f32x4, zero_union_mask);

    _mm_storeu_ps(results, result_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("popcnt")
#pragma clang attribute push(__attribute__((target("popcnt"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_hamming_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_u32_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    simsimd_u32_t differences = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        differences += _mm_popcnt_u64(*(simsimd_u64_t const *)a ^ *(simsimd_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b) differences += _mm_popcnt_u32(*a ^ *b);
    *result = differences;
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8_haswell(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n_words,
                                               simsimd_f32_t *result) {
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    simsimd_u32_t intersection_count = 0, union_count = 0;
    for (; n_words >= 8; n_words -= 8, a += 8, b += 8)
        intersection_count += (simsimd_u32_t)_mm_popcnt_u64(*(simsimd_u64_t const *)a & *(simsimd_u64_t const *)b),
            union_count += (simsimd_u32_t)_mm_popcnt_u64(*(simsimd_u64_t const *)a | *(simsimd_u64_t const *)b);
    for (; n_words; --n_words, ++a, ++b)
        intersection_count += simsimd_popcount_b8(*a & *b), union_count += simsimd_popcount_b8(*a | *b);
    *result = (union_count != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)union_count : 1.0f;
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *result) {
    simsimd_u32_t intersection_count = 0;
    simsimd_size_t n_remaining = n;
    for (; n_remaining >= 4; n_remaining -= 4, a += 4, b += 4) {
        __m128i a_u32x4 = _mm_loadu_si128((__m128i const *)a);
        __m128i b_u32x4 = _mm_loadu_si128((__m128i const *)b);
        __m128i equality_u32x4 = _mm_cmpeq_epi32(a_u32x4, b_u32x4);
        int equality_mask = _mm_movemask_ps(_mm_castsi128_ps(equality_u32x4));
        intersection_count += (simsimd_u32_t)_mm_popcnt_u32((unsigned int)equality_mask);
    }
    for (; n_remaining; --n_remaining, ++a, ++b) intersection_count += (*a == *b);
    *result = (n != 0) ? 1.0f - (simsimd_f32_t)intersection_count / (simsimd_f32_t)n : 1.0f;
}

/**
 *  @brief Running state for 256-bit Jaccard accumulation on Haswell.
 *
 *  This variant processes one 256-bit chunk at a time, matching Haswell's native
 *  AVX2 YMM register width. Unlike the b512 variant which used 2-way ILP for 8
 *  popcount operations, b256 uses a single accumulator since we process only 4 u64
 *  words per update. The out-of-order engine handles ILP across update calls.
 *
 *  @code{.c}
 *  // 512-bit binary vectors (2 x 256-bit chunks), one query and four targets
 *  simsimd_b8_t query[64], target_a[64], target_b[64], target_c[64], target_d[64];
 *  simsimd_f32_t query_popcount = 100.0f;
 *  simsimd_f32_t popcount_a = 95.0f, popcount_b = 110.0f, popcount_c = 88.0f, popcount_d = 102.0f;
 *
 *  simsimd_jaccard_b256_state_haswell_t state_a, state_b, state_c, state_d;
 *  simsimd_jaccard_b256_init_haswell(&state_a);
 *  simsimd_jaccard_b256_init_haswell(&state_b);
 *  simsimd_jaccard_b256_init_haswell(&state_c);
 *  simsimd_jaccard_b256_init_haswell(&state_d);
 *
 *  // Update for first 256-bit chunk
 *  __m256i query_chunk0 = _mm256_loadu_si256((__m256i const *)query);
 *  simsimd_jaccard_b256_update_haswell(&state_a, query_chunk0, _mm256_loadu_si256((__m256i const *)target_a));
 *  simsimd_jaccard_b256_update_haswell(&state_b, query_chunk0, _mm256_loadu_si256((__m256i const *)target_b));
 *  simsimd_jaccard_b256_update_haswell(&state_c, query_chunk0, _mm256_loadu_si256((__m256i const *)target_c));
 *  simsimd_jaccard_b256_update_haswell(&state_d, query_chunk0, _mm256_loadu_si256((__m256i const *)target_d));
 *
 *  // Update for second 256-bit chunk
 *  __m256i query_chunk1 = _mm256_loadu_si256((__m256i const *)(query + 32));
 *  simsimd_jaccard_b256_update_haswell(&state_a, query_chunk1, _mm256_loadu_si256((__m256i const *)(target_a + 32)));
 *  simsimd_jaccard_b256_update_haswell(&state_b, query_chunk1, _mm256_loadu_si256((__m256i const *)(target_b + 32)));
 *  simsimd_jaccard_b256_update_haswell(&state_c, query_chunk1, _mm256_loadu_si256((__m256i const *)(target_c + 32)));
 *  simsimd_jaccard_b256_update_haswell(&state_d, query_chunk1, _mm256_loadu_si256((__m256i const *)(target_d + 32)));
 *
 *  // Finalize all 4 states at once
 *  simsimd_f32_t results[4];
 *  simsimd_jaccard_b256_finalize_haswell(&state_a, &state_b, &state_c, &state_d,
 *      query_popcount, popcount_a, popcount_b, popcount_c, popcount_d, results);
 *  @endcode
 */
typedef struct simsimd_jaccard_b256_state_haswell_t {
    simsimd_u32_t intersection_count;
} simsimd_jaccard_b256_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_jaccard_b256_init_haswell(simsimd_jaccard_b256_state_haswell_t *state) {
    state->intersection_count = 0;
}

SIMSIMD_INTERNAL void simsimd_jaccard_b256_update_haswell(simsimd_jaccard_b256_state_haswell_t *state,
                                                          simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    // Process one 256-bit chunk (native Haswell AVX2 register size).
    //
    // Haswell port analysis:
    //   `_mm256_and_si256`:      p015, 1cy latency, 0.33cy throughput
    //   `_mm256_extracti128`:    p5, 3cy latency, 1cy throughput
    //   `_mm_cvtsi128_si64`:     p0, 2cy latency (first u64 extraction)
    //   `_mm_extract_epi64`:     p5, 3cy latency, 1cy throughput
    //   `_mm_popcnt_u64`:        p1 ONLY, 3cy latency, 1cy throughput (BOTTLENECK)
    //
    // With 4 popcounts per update, p1 is the bottleneck at 4 cycles minimum.

    // Step 1: Compute intersection bits (A AND B)
    __m256i intersection_u8x32 = _mm256_and_si256(a.ymm, b.ymm);

    // Step 2: Extract the two 128-bit halves
    __m128i intersection_low_u8x16 = _mm256_castsi256_si128(intersection_u8x32);       // FREE (register view)
    __m128i intersection_high_u8x16 = _mm256_extracti128_si256(intersection_u8x32, 1); // p5, 3cy

    // Step 3: Extract individual 64-bit words for scalar popcount
    simsimd_u64_t word_a = (simsimd_u64_t)_mm_cvtsi128_si64(intersection_low_u8x16);
    simsimd_u64_t word_b = (simsimd_u64_t)_mm_extract_epi64(intersection_low_u8x16, 1);
    simsimd_u64_t word_c = (simsimd_u64_t)_mm_cvtsi128_si64(intersection_high_u8x16);
    simsimd_u64_t word_d = (simsimd_u64_t)_mm_extract_epi64(intersection_high_u8x16, 1);

    // Step 4: Popcount each word (p1 bottleneck: 4 ops @ 1cy throughput = 4cy)
    simsimd_u32_t partial_a = (simsimd_u32_t)_mm_popcnt_u64(word_a);
    simsimd_u32_t partial_b = (simsimd_u32_t)_mm_popcnt_u64(word_b);
    simsimd_u32_t partial_c = (simsimd_u32_t)_mm_popcnt_u64(word_c);
    simsimd_u32_t partial_d = (simsimd_u32_t)_mm_popcnt_u64(word_d);

    // Step 5: Sum all partials (associative grouping for parallel adds)
    state->intersection_count += (partial_a + partial_b) + (partial_c + partial_d);
}

SIMSIMD_INTERNAL void simsimd_jaccard_b256_finalize_haswell(
    simsimd_jaccard_b256_state_haswell_t const *state_a, simsimd_jaccard_b256_state_haswell_t const *state_b,
    simsimd_jaccard_b256_state_haswell_t const *state_c, simsimd_jaccard_b256_state_haswell_t const *state_d,
    simsimd_f32_t query_popcount, simsimd_f32_t target_popcount_a, simsimd_f32_t target_popcount_b,
    simsimd_f32_t target_popcount_c, simsimd_f32_t target_popcount_d, simsimd_f32_t *results) {

    // 4-way SIMD Jaccard computation with fast reciprocal.
    //
    // Haswell port analysis:
    //   `_mm_setr_ps`:     p5, 1cy (INSERTPS chain)
    //   `_mm_add_ps`:      p01, 3cy latency
    //   `_mm_sub_ps`:      p01, 3cy latency
    //   `_mm_rcp_ps`:      p0, 5cy latency, 1cy throughput
    //   `_mm_mul_ps`:      p01, 5cy latency, 0.5cy throughput
    //   `_mm_blendv_ps`:   p015, 2cy latency

    // Pack intersection counts and convert to float
    simsimd_f32_t intersection_a_f32 = (simsimd_f32_t)state_a->intersection_count;
    simsimd_f32_t intersection_b_f32 = (simsimd_f32_t)state_b->intersection_count;
    simsimd_f32_t intersection_c_f32 = (simsimd_f32_t)state_c->intersection_count;
    simsimd_f32_t intersection_d_f32 = (simsimd_f32_t)state_d->intersection_count;

    __m128 intersection_f32x4 = _mm_setr_ps(intersection_a_f32, intersection_b_f32, intersection_c_f32,
                                            intersection_d_f32);
    __m128 query_f32x4 = _mm_set1_ps(query_popcount);
    __m128 targets_f32x4 = _mm_setr_ps(target_popcount_a, target_popcount_b, target_popcount_c, target_popcount_d);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case
    __m128 zero_union_mask = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_mask);

    // Fast reciprocal with Newton-Raphson refinement:
    //   `_mm_rcp_ps`: ~12-bit precision, 5cy latency, 1cy throughput
    //   Newton-Raphson: rcp' = rcp * (2 - x * rcp), doubles precision to ~22-24 bits
    // Total: ~10cy vs `_mm_div_ps` 13cy latency, but NR has better throughput
    __m128 union_reciprocal_f32x4 = _mm_rcp_ps(safe_union_f32x4);
    __m128 newton_raphson_correction = _mm_sub_ps(two_f32x4, _mm_mul_ps(safe_union_f32x4, union_reciprocal_f32x4));
    union_reciprocal_f32x4 = _mm_mul_ps(union_reciprocal_f32x4, newton_raphson_correction);

    __m128 ratio_f32x4 = _mm_mul_ps(intersection_f32x4, union_reciprocal_f32x4);
    __m128 jaccard_f32x4 = _mm_sub_ps(one_f32x4, ratio_f32x4);
    __m128 result_f32x4 = _mm_blendv_ps(jaccard_f32x4, one_f32x4, zero_union_mask);

    _mm_storeu_ps(results, result_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_hamming_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n,
                                       simsimd_u32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_hamming_b8_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_hamming_b8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_hamming_b8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_hamming_b8_haswell(a, b, n, result);
#else
    simsimd_hamming_b8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_jaccard_b8(simsimd_b8_t const *a, simsimd_b8_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_jaccard_b8_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_jaccard_b8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_jaccard_b8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_jaccard_b8_haswell(a, b, n, result);
#else
    simsimd_jaccard_b8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_jaccard_u32(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_jaccard_u32_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_jaccard_u32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_jaccard_u32_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_jaccard_u32_haswell(a, b, n, result);
#else
    simsimd_jaccard_u32_serial(a, b, n, result);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
