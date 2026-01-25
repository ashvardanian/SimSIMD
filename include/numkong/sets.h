/**
 *  @brief Batched pairwise binary set computations (Hamming and Jaccard distances).
 *  @file include/numkong/sets.h
 *  @author Ash Vardanian
 *
 *  This module provides efficient batched computation of Hamming and Jaccard distances
 *  between large collections of sets. Unlike the single-vector `set.h` module, this module
 *  is optimized for matrix-style operations where you compute distances between:
 *
 *  - All pairs of rows in a query matrix Q against rows in values matrix V
 *  - All pairs within a single matrix values matrix V (symmetric kernel)
 *
 *  @section sets_use_cases Use Cases
 *
 *  - Binary similarity search: Find nearest neighbors in Hamming/Jaccard space
 *  - MinHash/SimHash: Compute Jaccard similarity for document fingerprints
 *  - Locality-sensitive hashing (LSH): Build similarity graphs
 *  - Binary neural network inference: Compute distances for BNN outputs
 *
 *  @section sets_math Mathematical Background
 *
 *  Hamming distance: Number of positions where bits differ
 *    hamming(a, b) = popcount(a XOR b)
 *
 *  Jaccard distance: 1 minus the Jaccard similarity
 *    jaccard(a, b) = 1 - |a ∩ b| / |a ∪ b|
 *                  = 1 - popcount(a AND b) / popcount(a OR b)
 *
 *  For Jaccard, we use the identity: |a ∪ b| = |a| + |b| - |a ∩ b|
 *  This allows precomputing |a| and |b| (population counts) during packing.
 */

#ifndef NK_SETS_H
#define NK_SETS_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Calculate buffer size needed for packing the query matrix for Hamming distances.
 *  @param[in] n Number of rows in the query matrix
 *  @param[in] d Number of dimensions in each row vector
 *  @return Required buffer size in bytes
 */
NK_DYNAMIC nk_size_t nk_hammings_packed_size_u1(nk_size_t n, nk_size_t d);
/** @copydoc nk_hammings_packed_size_u1 */
NK_DYNAMIC nk_size_t nk_hammings_packed_size_u8(nk_size_t n, nk_size_t d);

/**
 *  @brief Calculate buffer size needed for packing the query matrix for Jaccard distances.
 *  @param[in] n Number of rows in the query matrix
 *  @param[in] d Number of dimensions in each row vector
 *  @return Required buffer size in bytes
 */
NK_DYNAMIC nk_size_t nk_jaccards_packed_size_u1(nk_size_t n, nk_size_t d);
/** @copydoc nk_hammings_packed_size_u32 */
NK_DYNAMIC nk_size_t nk_jaccards_packed_size_u32(nk_size_t n, nk_size_t d);

/**
 *  @brief Pack the queries matrix for efficient Hamming distance computation.
 *  @param[in] q Row major queries matrix with continuous rows, separated a given stride
 *  @param[in] n Number of rows in the queries matrix
 *  @param[in] d Number of bits per vector
 *  @param[in] q_stride Byte stride between rows of the query matrix
 *  @param[out] q_packed Output buffer
 */
NK_DYNAMIC void nk_hammings_pack_u1(nk_u1x8_t const *q, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed);
/** @copydoc nk_hammings_pack_u1 */
NK_DYNAMIC void nk_hammings_pack_u8(nk_u1x8_t const *q, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed);

/**
 *  @brief Pack the queries matrix for efficient Jaccard distance computation, precomputing vector magnitudes.
 *  @param[in] q Row major queries matrix with continuous rows, separated a given stride
 *  @param[in] n Number of rows in the queries matrix
 *  @param[in] d Number of bits per vector
 *  @param[in] q_stride Byte stride between rows of the query matrix
 *  @param[out] q_packed Output buffer
 */
NK_DYNAMIC void nk_jaccards_pack_u1(nk_u1x8_t const *q, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed);
/** @copydoc nk_jaccards_pack_u1 */
NK_DYNAMIC void nk_jaccards_pack_u32(nk_u1x8_t const *q, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed);

/**
 *  @brief Compute Hamming distances between V rows and packed Q rows.
 *  @param[in] v Input values matrix
 *  @param[in] q_packed Packed queries matrix
 *  @param[out] result Row-major results matrix
 *  @param[in] rows Number of rows in the results matrix
 *  @param[in] cols Number of columns in the results matrix
 *  @param[in] d Number of dimensions (depth) per vector
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 */
NK_PUBLIC void nk_hammings_packed_u1(nk_u1x8_t const *a, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u8(nk_u8_t const *a, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes);

/**
 *  @brief Compute Jaccard distances between V rows and packed Q rows.
 *  @param[in] v Input values matrix
 *  @param[in] q_packed Packed queries matrix
 *  @param[out] result Row-major results matrix
 *  @param[in] rows Number of rows in the results matrix
 *  @param[in] cols Number of columns in the results matrix
 *  @param[in] d Number of dimensions (depth) per vector
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 */
NK_PUBLIC void nk_jaccards_packed_u1(nk_u1x8_t const *a, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u32(nk_u32_t const *a, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);
/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix of Hamming distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] n_vectors Number of vectors (rows) in the input matrix.
 *  @param[in] d Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (n_vectors × n_vectors).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 */
NK_PUBLIC void nk_hammings_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n);
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_symmetric_u8(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n);
/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix of Jaccard distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] n_vectors Number of vectors (rows) in the input matrix.
 *  @param[in] d Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (n_vectors × n_vectors).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 */
NK_PUBLIC void nk_jaccards_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_hammings_symmetric_u32(nk_u32_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                         nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n);

#include "numkong/sets/serial.h"
#include "numkong/sets/smebi32.h"
#include "numkong/sets/neon.h"
#include "numkong/sets/ice.h"
#include "numkong/sets/haswell.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_hammings_packed_size_u1(nk_size_t n, nk_size_t d) {
#if NK_TARGET_SMEBI32
    return nk_hammings_packed_size_u1_smebi32(n, d);
#elif NK_TARGET_NEON
    return nk_hammings_packed_size_u1_neon(n, d);
#elif NK_TARGET_ICE
    return nk_hammings_packed_size_u1_ice(n, d);
#elif NK_TARGET_HASWELL
    return nk_hammings_packed_size_u1_haswell(n, d);
#else
    return nk_hammings_packed_size_u1_serial(n, d);
#endif
}

NK_PUBLIC void nk_hammings_pack_u1(nk_u1x8_t const *b, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed) {
#if NK_TARGET_SMEBI32
    nk_hammings_pack_u1_smebi32(b, n, d, q_stride, q_packed);
#elif NK_TARGET_NEON
    nk_hammings_pack_u1_neon(b, n, d, q_stride, q_packed);
#elif NK_TARGET_ICE
    nk_hammings_pack_u1_ice(b, n, d, q_stride, q_packed);
#elif NK_TARGET_HASWELL
    nk_hammings_pack_u1_haswell(b, n, d, q_stride, q_packed);
#else
    nk_hammings_pack_u1_serial(b, n, d, q_stride, q_packed);
#endif
}

NK_PUBLIC void nk_hammings_packed_u1(nk_u1x8_t const *a, void const *q_packed, nk_u32_t *result, nk_size_t n,
                                     nk_size_t column_count, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEBI32
    nk_hammings_packed_u1_smebi32(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_hammings_packed_u1_neon(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICE
    nk_hammings_packed_u1_ice(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_hammings_packed_u1_haswell(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes);
#else
    nk_hammings_pack_u1_serial(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes);
#endif
}

NK_PUBLIC void nk_hammings_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n) {
#if NK_TARGET_SMEBI32
    nk_hammings_symmetric_u1_smebi32(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_NEON
    nk_hammings_symmetric_u1_neon(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_ICE
    nk_hammings_symmetric_u1_ice(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_HASWELL
    nk_hammings_symmetric_u1_haswell(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#else
    nk_hammings_symmetric_u1_serial(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#endif
}

NK_PUBLIC nk_size_t nk_jaccards_packed_size_u1(nk_size_t n, nk_size_t d) {
#if NK_TARGET_SMEBI32
    return nk_jaccards_packed_size_u1_smebi32(n, d);
#elif NK_TARGET_NEON
    return nk_jaccards_packed_size_u1_neon(n, d);
#elif NK_TARGET_ICE
    return nk_jaccards_packed_size_u1_ice(n, d);
#elif NK_TARGET_HASWELL
    return nk_jaccards_packed_size_u1_haswell(n, d);
#else
    return nk_jaccards_packed_size_u1_serial(n, d);
#endif
}

NK_PUBLIC void nk_jaccards_pack_u1(nk_u1x8_t const *b, nk_size_t n, nk_size_t d, nk_size_t q_stride, void *q_packed) {
#if NK_TARGET_SMEBI32
    nk_jaccards_pack_u1_smebi32(b, n, d, q_stride, q_packed);
#elif NK_TARGET_NEON
    nk_jaccards_pack_u1_neon(b, n, d, q_stride, q_packed);
#elif NK_TARGET_ICE
    nk_jaccards_pack_u1_ice(b, n, d, q_stride, q_packed);
#elif NK_TARGET_HASWELL
    nk_jaccards_pack_u1_haswell(b, n, d, q_stride, q_packed);
#else
    nk_jaccards_pack_u1_serial(b, n, d, q_stride, q_packed);
#endif
}

NK_PUBLIC void nk_jaccards_packed_u1(nk_u1x8_t const *a, void const *q_packed, nk_f32_t *result, nk_size_t n,
                                     nk_size_t column_count, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes, nk_f32_t const *a_norms) {
#if NK_TARGET_SMEBI32
    nk_jaccards_packed_u1_smebi32(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes,
                                  a_norms);
#elif NK_TARGET_NEON
    nk_jaccards_packed_u1_neon(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes, a_norms);
#elif NK_TARGET_ICE
    nk_jaccards_packed_u1_ice(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes, a_norms);
#elif NK_TARGET_HASWELL
    nk_jaccards_packed_u1_haswell(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes,
                                  a_norms);
#else
    nk_jaccards_packed_u1_serial(a, q_packed, result, n, column_count, d, v_stride_in_bytes, r_stride_in_bytes,
                                 a_norms);
#endif
}

NK_PUBLIC void nk_jaccards_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t n) {
#if NK_TARGET_SMEBI32
    nk_jaccards_symmetric_u1_smebi32(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_NEON
    nk_jaccards_symmetric_u1_neon(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_ICE
    nk_jaccards_symmetric_u1_ice(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#elif NK_TARGET_HASWELL
    nk_jaccards_symmetric_u1_haswell(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#else
    nk_jaccards_symmetric_u1_serial(vectors, n_vectors, d, stride, result, result_stride, row_start, n);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SETS_H
