/**
 *  @brief SIMD-accelerated Binary Similarity Measures optimized for SIMD-free CPUs.
 *  @file include/numkong/binary/serial.h
 *  @sa include/numkong/binary.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_BINARY_SERIAL_H
#define NK_BINARY_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC unsigned char nk_popcount_b8(nk_b8_t x) {
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

NK_PUBLIC void nk_hamming_b8_serial(nk_b8_t const *a, nk_b8_t const *b, nk_size_t n_words, nk_u32_t *result) {
    nk_u32_t differences = 0;
    for (nk_size_t i = 0; i != n_words; ++i) differences += nk_popcount_b8(a[i] ^ b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_b8_serial(nk_b8_t const *a, nk_b8_t const *b, nk_size_t n_words, nk_f32_t *result) {
    nk_u32_t intersection_count = 0, union_count = 0;
    for (nk_size_t i = 0; i != n_words; ++i)
        intersection_count += nk_popcount_b8(a[i] & b[i]), union_count += nk_popcount_b8(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 1.0f;
}

NK_PUBLIC void nk_jaccard_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    for (nk_size_t i = 0; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 1.0f;
}

typedef struct nk_jaccard_b128_state_serial_t {
    nk_u64_t intersection_count;
} nk_jaccard_b128_state_serial_t;

NK_INTERNAL void nk_jaccard_b128_init_serial(nk_jaccard_b128_state_serial_t *state) { state->intersection_count = 0; }

NK_INTERNAL void nk_jaccard_b128_update_serial(nk_jaccard_b128_state_serial_t *state, nk_b128_vec_t a,
                                               nk_b128_vec_t b) {
    nk_u64_t intersection_low = a.u64s[0] & b.u64s[0];
    nk_u64_t intersection_high = a.u64s[1] & b.u64s[1];
    state->intersection_count += nk_u64_popcount_(intersection_low);
    state->intersection_count += nk_u64_popcount_(intersection_high);
}

NK_INTERNAL void nk_jaccard_b128_finalize_serial(nk_jaccard_b128_state_serial_t const *state_a,
                                                 nk_jaccard_b128_state_serial_t const *state_b,
                                                 nk_jaccard_b128_state_serial_t const *state_c,
                                                 nk_jaccard_b128_state_serial_t const *state_d, nk_f32_t query_popcount,
                                                 nk_f32_t target_popcount_a, nk_f32_t target_popcount_b,
                                                 nk_f32_t target_popcount_c, nk_f32_t target_popcount_d,
                                                 nk_f32_t *results) {

    nk_f32_t intersection_a = (nk_f32_t)state_a->intersection_count;
    nk_f32_t intersection_b = (nk_f32_t)state_b->intersection_count;
    nk_f32_t intersection_c = (nk_f32_t)state_c->intersection_count;
    nk_f32_t intersection_d = (nk_f32_t)state_d->intersection_count;

    nk_f32_t union_a = query_popcount + target_popcount_a - intersection_a;
    nk_f32_t union_b = query_popcount + target_popcount_b - intersection_b;
    nk_f32_t union_c = query_popcount + target_popcount_c - intersection_c;
    nk_f32_t union_d = query_popcount + target_popcount_d - intersection_d;

    results[0] = (union_a != 0) ? 1.0f - intersection_a / union_a : 1.0f;
    results[1] = (union_b != 0) ? 1.0f - intersection_b / union_b : 1.0f;
    results[2] = (union_c != 0) ? 1.0f - intersection_c / union_c : 1.0f;
    results[3] = (union_d != 0) ? 1.0f - intersection_d / union_d : 1.0f;
}

#if defined(__cplusplus)
} // extern "C"
#endif