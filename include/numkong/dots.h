/**
 *  @brief SIMD-accelerated Batched Dot Products.
 *  @file include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *
 *  Implements batch dot-product kernels computing C[m × n] = A[m × k] × B[n × k]ᵀ
 *  with row-major A and arbitrary B, optimized for ML inference and similarity workloads.
 *
 *  Primary Use Cases (1-to-N focus):
 *
 *  - k-NN search: ‖a-b‖² = ‖a‖² + ‖b‖² - 2(a × b)
 *  - Cosine similarity: (a × b) / (‖a‖ × ‖b‖)
 *  - Sparse attention patterns
 *  - Embedding similarity matrices
 *  - k-means clustering, DBSCAN, hierarchical clustering
 *
 *  It implements seveal operations:
 *
 *  - "dots_packed" - computing dot-products where the B matrix is pre-packed into optimal form
 *  - "dots_packed_size" - which estimates the memory requrements for external `malloc`
 *  - "dots_pack" - to perform the pre-preocessing
 *  - "dots_compact" - optional helpers to normalize or downcast into original precision
 *  - "dots_symmetric" - for A × Aᵀ Gram matrix multiplication
 *
 *  If the original "dots_packed" is analogous to "GEMM" (General Matrix Multiplication) in BLAS,
 *  the "dots_symmetric" is similar to the "SYRK" (the Symmetric rank-k update of a matrix).
 *
 *  Following numeric types are prioritized:
 *
 *  - `f32`: Float32 inputs accumulating to Float64, downcasting into Float32 in the end
 *  - `bf16`: BFloat16 inputs accumulating to Float32
 *  - `i8`: Int8 inputs accumulating to Int32
 *  - `e4m3` & `e5m2`: Float8 inputs often leverating BFloat16 products, accumulating to Float32
 *
 *  For hardware architectures:
 *
 *  - x86: Haswell (AVX2), Genoa (AVX512-BF16), Sapphire Rapids (AMX)
 *  - Arm: NEON, SVE, SME
 *
 *  @section memory_layout Memory Layout and Transpose Semantics
 *
 *  All matrices use row-major storage. Column-major is NOT supported.
 *  The kernel computes C = A × Bᵀ where:
 *
 *  - A is (m × k): m rows, k columns, stride = a_stride bytes between rows
 *  - B is (n × k): n rows, k columns, stride = b_stride bytes between rows
 *  - C is (m × n): m rows, n columns, stride = c_stride bytes between rows
 *
 *  This means C[i,j] = dot(row i of A, row j of B) = Σₗ A[i,l] × B[j,l].
 *
 *  All strides are in bytes.
 *
 *  To compute standard A × B (where B is k × n), pass Bᵀ to the packing function:
 *
 *  @code{.c}
 *  // Standard matmul: C[m × n] = A[m × k] × B[k × n]
 *  // B is stored row-major as k rows of n elements
 *  // Treat it as Bᵀ: n rows of k elements with stride = sizeof(element)
 *  nk_dots_pack_bf16(b, n, k, sizeof(nk_bf16_t), b_packed);
 *  nk_dots_packed_bf16(a, b_packed, c, m, n, k, a_stride, c_stride);
 *  // Result: C = A × (Bᵀ)ᵀ = A × B
 *  @endcode
 *
 *  @section two_phase_api Two-Phase API for Static Weights
 *
 *  Matrix multiplication hardware (AMX, SME) requires specific data layouts that differ
 *  from standard row-major ordering. Since one matrix (typically weights in neural networks)
 *  is often static, we provide a two-phase API: pack once, multiply many times.
 *
 *  @code{.c}
 *  // Similarity search: C[m × n] = queries[m × k] × database[n × k]ᵀ
 *  // Both matrices stored row-major, each row is one vector of dimension k
 *  nk_size_t packed_bytes = nk_dots_packed_size_bf16(n, k);
 *  void *b_packed = malloc(packed_bytes);
 *  nk_dots_pack_bf16(database, n, k, k * sizeof(nk_bf16_t), b_packed);
 *  nk_dots_packed_bf16(queries, b_packed, c, m, n, k, ...);
 *  // Result: C[i,j] = dot(query i, database vector j)
 *  @endcode
 *
 *  The packed format is opaque and backend-specific. AMX expects (16 × 32) tiles with interleaved
 *  pairs, while NEON/SVE use arrangements optimized for their vector lengths.
 *
 *  @section why_int8 Why INT8 and Not UINT8?
 *
 *  Unsigned 8-bit integers were considered but deprioritized. The industry has converged on
 *  signed INT8 as the standard for quantized inference:
 *
 *      Framework           Default     Notes
 *      PyTorch             qint8       New X86 backend uses INT8 via oneDNN
 *      TensorFlow Lite     int8        Actively removing UINT8 support
 *      ONNX Runtime        S8S8        "Should be the first choice"
 *      TensorRT            INT8        Symmetric [-128,127], no UINT8 option
 *      ARM CMSIS-NN        int8        Follows TFLite INT8 spec exactly
 *
 *  @section why_no_scaling Why No Alpha/Beta Scaling?
 *
 *  BLAS-style `C = α × A × B + β × C` scaling was considered but omitted. While useful for scientific
 *  computing (iterative solvers, matrix factorizations), it's rarely used in ML inference where
 *  frameworks handle such operations via graph fusion. More importantly, on chips with separate
 *  physical registers for vector and matrix operations (like AMX), moving scalars between register
 *  files adds transfer latency that negates any benefit.
 *
 *  @section why_no_pad Why Not Pad N Dimension to Eliminate Edge Handling?
 *
 *  Padding N to a tile-aligned boundary (multiple of 16) during packing was considered to eliminate
 *  the separate AVX-512 edge kernel for N remainder rows. While this sounds simpler ("pure AMX"),
 *  it actually increases code size by ~125 lines because:
 *
 *  - The AVX-512 edge fallback is compact (~40 lines) and handles both full-M × N-edge and
 *    M-edge × N-edge cases through a single reusable function
 *  - Replacing it with "AMX + masked stores" requires verbose tile handling code duplicated
 *    across all 4 multiply functions (aligned/misaligned × BF16/I8)
 *  - Each function needs a new "trailing N tile for full M blocks" section (~50 lines each)
 *
 *  The current hybrid layout (AMX for full tiles, AVX-512 for edges) is more maintainable despite
 *  being conceptually less uniform. Memory overhead of the edge region is negligible (<2% worst case).
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Low-precision matmul relies on VPMADD* (AVX2), VNNI dot-products, and BF16 dot-products
 *  on AVX-512. Zen4 improves throughput by dual-issuing many integer ops on FP ports.
 *
 *      Intrinsic                     Instruction                       Haswell     Genoa
 *      _mm256_maddubs_epi16          VPMADDUBSW (YMM, YMM, YMM)         5c @ p0     3c @ p01
 *      _mm256_madd_epi16             VPMADDWD (YMM, YMM, YMM)           5c @ p0     3c @ p01
 *      _mm256_dpbusd_epi32           VPDPBUSD (YMM, K, YMM, YMM)        n/a         4c @ p01
 *      _mm256_dpwssds_epi32          VPDPWSSDS (YMM, K, YMM, YMM)       n/a         4c @ p01
 *      _mm256_dpbf16_ps              VDPBF16PS (YMM, YMM, YMM)          n/a         6c @ p01
 *
 *  AMX tile ops (TDPBF16PS/TDPBUSD/TDPBSSD) are not covered by the uops.info 2022 dataset.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - uops.info: https://uops.info/
 *  - Matrix Multiplication in 40 lines: https://en.algorithmica.org/hpc/algorithms/matmul/
 *  - LLaMA CPU optimization: https://justine.lol/matmul/
 *  - SME outer-product notes: https://github.com/tzakharko/m4-sme-exploration
 *
 */
#ifndef NK_DOTS_H
#define NK_DOTS_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for second multiplier matrix (B).
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_bf16(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f16(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e4m3(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e5m2(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e2m3(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e3m2(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f32(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f64(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_i8(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_u8(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_i4(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_u4(nk_size_t n, nk_size_t k);

/**
 *  @brief Packs the second multiplier (B) matrix into a backend-specific layout.
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from nk_dots_packed_size_bf16.
 */
NK_DYNAMIC void nk_dots_pack_bf16(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f16(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e4m3(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e5m2(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e2m3(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e3m2(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f32(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f64(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_i8(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_u8(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_i4(nk_i4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_u4(nk_u4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed second multiplier matrix (B), accumulating into C.
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced.
 *  @param[out] c The output C matrix in row-major order.
 *  @param[in] m The number of rows in A.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
NK_DYNAMIC void nk_dots_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_i8(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_u8(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);

/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] n_vectors Number of vectors (rows) in the input matrix.
 *  @param[in] depth Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (n_vectors × n_vectors).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 */
NK_DYNAMIC void nk_dots_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                       nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f16(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                       nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                       nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                       nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                       nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f32(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_i8(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_u8(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);

/**
 *  @brief Compacts f32 GEMM output to bf16 (in-place).
 *
 *  After computing C_f32 = A × Bᵀ in f32, truncates to bf16 with rounding.
 *  The operation is done in-place: reads f32 values and writes bf16 to the same buffer.
 *  Output is tightly packed with stride = n × sizeof(bf16).
 *
 *  @param c Buffer containing f32 values, will be overwritten with bf16 output (m × n).
 *  @param m Number of rows.
 *  @param n Number of columns.
 *  @param c_stride Row stride of input f32 matrix in bytes.
 */
NK_DYNAMIC void nk_dots_compact_bf16(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);

/**
 *  @brief Compacts i32 GEMM output to normalized i8 (in-place).
 *
 *  After computing C_i32 = A × Bᵀ in i32, normalizes to cosine similarity in [-128, 127].
 *  Uses squared norms for normalization: result[i,j] = 127 × C[i,j] / sqrt(a_norm[i] × b_norm[j]).
 *  The operation is done in-place: reads i32 values and writes i8 to the same buffer.
 *  Output is tightly packed with stride = n × sizeof(i8).
 *
 *  @param c Buffer containing i32 values, will be overwritten with i8 output (m × n).
 *  @param m Number of rows.
 *  @param n Number of columns.
 *  @param c_stride Row stride of input i32 matrix in bytes.
 *  @param a_squared_norms Squared L2 norms for A rows (length m).
 *  @param b_squared_norms Squared L2 norms for B rows (length n).
 */
NK_DYNAMIC void nk_dots_compact_i8(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                   nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);

/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_serial(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_serial(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_serial(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_serial(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_compact_bf16 */
NK_PUBLIC void nk_dots_compact_bf16_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_serial(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_compact_i8 */
NK_PUBLIC void nk_dots_compact_i8_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                         nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_serial(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_serial(nk_u4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_serial(nk_i4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_serial(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_serial(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products.
 *  Packing interleaves elements for SIMD broadcast patterns.
 */
#if NK_TARGET_GENOA
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_genoa(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_compact_bf16 */
NK_PUBLIC void nk_dots_compact_bf16_genoa(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_genoa(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_genoa(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_genoa(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_genoa(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_genoa(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_genoa(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);

/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_genoa(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_genoa(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_GENOA

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16 × 32) BF16 or (16 × 64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sapphireamx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_sapphireamx(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                               nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_compact_bf16 */
NK_PUBLIC void nk_dots_compact_bf16_sapphireamx(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sapphireamx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sapphireamx(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m,
                                             nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_compact_i8 */
NK_PUBLIC void nk_dots_compact_i8_sapphireamx(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                              nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sapphireamx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_sapphireamx(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                               nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);

/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sapphireamx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_sapphireamx(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                               nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
#endif // NK_TARGET_SAPPHIREAMX

/*  ARM SME backends using Scalable Matrix Extension.
 *  SME provides ZA tile registers for outer product operations.
 *  F16/BF16/I8/U8/E4M3 use ZA32 tiles, F32/F64 use ZA64 tiles (FEAT_SME_F64F64).
 */
#if NK_TARGET_SME
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_sme(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_sme(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sme(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_sme(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_sme(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_sme(nk_u4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_sme(nk_i4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SME

/*  ARM SME with FEAT_SME_F64F64 (F32/F64 with F64 accumulators).
 *  Requires Apple M4 or equivalent with F64 outer product support.
 */
#if NK_TARGET_SMEF64
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_smef64(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SMEF64

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_haswell(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_haswell(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_haswell(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_haswell(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_haswell(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_haswell(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_haswell(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_haswell(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_haswell(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_haswell(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_HASWELL

/*  Skylake backends using AVX-512 (Intel Core 6th gen+).
 *  Provides 512-bit vectors (16× f32, 8× f64), supporting F32/F64/F16/BF16/FP8 with FMA.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_skylake(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_skylake(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_skylake(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_skylake(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_skylake(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_skylake(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_skylake(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_skylake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_skylake(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SKYLAKE

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products.
 */
#if NK_TARGET_ICELAKE
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_icelake(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_icelake(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                         nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_icelake(nk_i4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_icelake(nk_u4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ICELAKE

/*  Sierra backends using AVX10.2 with VMPSADBW.
 *  Optimized for I8/U8 via VMPSADBW (vector multiply-sum of absolute differences).
 */
#if NK_TARGET_SIERRA
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sierra(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sierra(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sierra(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_sierra(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SIERRA

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64.
 */
#if NK_TARGET_NEON
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_neon(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_neon(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_neon(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_neon(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEON

/*  ARM NEON with F16 arithmetic (ARMv8.2-A FP16).
 *  Provides native F16 FMLA for half-precision dot products.
 */
#if NK_TARGET_NEONHALF
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_neonhalf(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_neonhalf(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_neonhalf(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_neonhalf(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONHALF

/*  ARM NEON with BF16 dot product (ARMv8.6-A BF16).
 *  Uses BFDOT/BFMMLA for efficient BF16 matrix operations.
 */
#if NK_TARGET_NEONBFDOT
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_neonbfdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_neonbfdot(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                             nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONBFDOT

/*  ARM NEON with signed/unsigned dot product (ARMv8.2-A DotProd).
 *  Provides SDOT/UDOT for I8/U8 vector dot products.
 */
#if NK_TARGET_NEONSDOT
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_neonsdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_neonsdot(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                          nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_neonsdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_neonsdot(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                          nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONSDOT

/*  ARM NEON with FP16 FML (fused multiply-long, ARMv8.2-A FP16FML).
 *  Uses FMLAL/FMLSL for F16 and custom FP8 (E2M3/E3M2) operations.
 */
#if NK_TARGET_NEONFHM
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_neonfhm(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_neonfhm(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_neonfhm(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_neonfhm(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_neonfhm(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_neonfhm(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_neonfhm(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_neonfhm(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_neonfhm(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                           nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_neonfhm(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONFHM

/**
 *  @brief  Returns the output dtype for dot products.
 */
NK_INTERNAL nk_dtype_t nk_dots_packed_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_i8_k: return nk_i32_k;
    case nk_u8_k: return nk_u32_k;
    case nk_u1_k: return nk_u32_k;
    case nk_u4_k: return nk_u32_k;
    case nk_i4_k: return nk_i32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/dots/serial.h"
#include "numkong/dots/haswell.h"
#include "numkong/dots/skylake.h"
#include "numkong/dots/icelake.h"
#include "numkong/dots/sierra.h"
#include "numkong/dots/genoa.h"
#include "numkong/dots/sapphireamx.h"
#include "numkong/dots/neon.h"
#include "numkong/dots/neonsdot.h"
#include "numkong/dots/neonhalf.h"
#include "numkong/dots/neonfhm.h"
#include "numkong/dots/neonbfdot.h"
#include "numkong/dots/sve.h"
#include "numkong/dots/svehalf.h"
#include "numkong/dots/sme.h"
#include "numkong/dots/smef64.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_dots_packed_size_f32(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SMEF64
    return nk_dots_packed_size_f32_smef64(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f32_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f32_haswell(n, k);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_f32_neon(n, k);
#else
    return nk_dots_packed_size_f32_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_f32(nk_f32_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SMEF64
    nk_dots_pack_f32_smef64(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f32_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f32_haswell(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_f32_neon(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_f32_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SMEF64
    nk_dots_packed_f32_smef64(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f32_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f32_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_f32_neon(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_f32_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_f64(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SMEF64
    return nk_dots_packed_size_f64_smef64(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f64_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f64_haswell(n, k);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_f64_neon(n, k);
#else
    return nk_dots_packed_size_f64_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_f64(nk_f64_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SMEF64
    nk_dots_pack_f64_smef64(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f64_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f64_haswell(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_f64_neon(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_f64_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SMEF64
    nk_dots_packed_f64_smef64(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f64_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f64_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_f64_neon(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_f64_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_f16(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_f16_sme(n, k);
#elif NK_TARGET_NEONFHM
    return nk_dots_packed_size_f16_neonfhm(n, k);
#elif NK_TARGET_NEONHALF
    return nk_dots_packed_size_f16_neonhalf(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f16_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f16_haswell(n, k);
#else
    return nk_dots_packed_size_f16_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_f16(nk_f16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_f16_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONFHM
    nk_dots_pack_f16_neonfhm(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONHALF
    nk_dots_pack_f16_neonhalf(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f16_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f16_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_f16_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                  nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_f16_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONFHM
    nk_dots_packed_f16_neonfhm(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONHALF
    nk_dots_packed_f16_neonhalf(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f16_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f16_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_f16_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_bf16_sme(n, k);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_bf16_sapphireamx(n, k);
#elif NK_TARGET_NEONBFDOT
    return nk_dots_packed_size_bf16_neonbfdot(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_bf16_genoa(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_bf16_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_bf16_haswell(n, k);
#else
    return nk_dots_packed_size_bf16_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_bf16(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_bf16_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_bf16_sapphireamx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONBFDOT
    nk_dots_pack_bf16_neonbfdot(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_bf16_genoa(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_bf16_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_bf16_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_bf16_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_bf16_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_bf16_sapphireamx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONBFDOT
    nk_dots_packed_bf16_neonbfdot(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_bf16_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_bf16_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_bf16_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_bf16_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_compact_bf16(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride) {
#if NK_TARGET_SAPPHIREAMX
    nk_dots_compact_bf16_sapphireamx(c, m, n, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_compact_bf16_genoa(c, m, n, c_stride);
#else
    nk_dots_compact_bf16_serial(c, m, n, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_i8(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_i8_sme(n, k);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_i8_sapphireamx(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_i8_neonsdot(n, k);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_i8_icelake(n, k);
#elif NK_TARGET_SIERRA
    return nk_dots_packed_size_i8_sierra(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_i8_haswell(n, k);
#else
    return nk_dots_packed_size_i8_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_i8(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_i8_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_i8_sapphireamx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_i8_neonsdot(b, n, k, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_i8_icelake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SIERRA
    nk_dots_pack_i8_sierra(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_i8_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_i8_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_i8(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                 nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_i8_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_i8_sapphireamx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_i8_neonsdot(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_i8_icelake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SIERRA
    nk_dots_packed_i8_sierra(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_i8_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_i8_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_compact_i8(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                  nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms) {
#if NK_TARGET_SAPPHIREAMX
    nk_dots_compact_i8_sapphireamx(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#else
    nk_dots_compact_i8_serial(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u8(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_u8_sme(n, k);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_u8_sapphireamx(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_u8_neonsdot(n, k);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_u8_icelake(n, k);
#elif NK_TARGET_SIERRA
    return nk_dots_packed_size_u8_sierra(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_u8_haswell(n, k);
#else
    return nk_dots_packed_size_u8_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_u8(nk_u8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_u8_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_u8_sapphireamx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_u8_neonsdot(b, n, k, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_u8_icelake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SIERRA
    nk_dots_pack_u8_sierra(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_u8_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_u8_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_u8(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                 nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_u8_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_u8_sapphireamx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_u8_neonsdot(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_u8_icelake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SIERRA
    nk_dots_packed_u8_sierra(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_u8_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_u8_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e4m3_sme(n, k);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e4m3_sapphireamx(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e4m3_genoa(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e4m3_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e4m3_haswell(n, k);
#else
    return nk_dots_packed_size_e4m3_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_e4m3(nk_e4m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e4m3_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e4m3_sapphireamx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e4m3_genoa(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e4m3_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e4m3_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_e4m3_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e4m3_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e4m3_sapphireamx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e4m3_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e4m3_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e4m3_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_e4m3_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e5m2_sme(n, k);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e5m2_sapphireamx(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e5m2_genoa(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e5m2_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e5m2_haswell(n, k);
#else
    return nk_dots_packed_size_e5m2_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_e5m2(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e5m2_sme(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e5m2_sapphireamx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e5m2_genoa(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e5m2_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e5m2_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_e5m2_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e5m2_sme(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e5m2_sapphireamx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e5m2_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e5m2_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e5m2_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_e5m2_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3(nk_size_t n, nk_size_t k) {
#if NK_TARGET_NEONFHM
    return nk_dots_packed_size_e2m3_neonfhm(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e2m3_genoa(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e2m3_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e2m3_haswell(n, k);
#else
    return nk_dots_packed_size_e2m3_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_e2m3(nk_e2m3_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_NEONFHM
    nk_dots_pack_e2m3_neonfhm(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e2m3_genoa(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e2m3_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e2m3_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_e2m3_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_NEONFHM
    nk_dots_packed_e2m3_neonfhm(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e2m3_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e2m3_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e2m3_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_e2m3_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2(nk_size_t n, nk_size_t k) {
#if NK_TARGET_NEONFHM
    return nk_dots_packed_size_e3m2_neonfhm(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e3m2_genoa(n, k);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e3m2_skylake(n, k);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e3m2_haswell(n, k);
#else
    return nk_dots_packed_size_e3m2_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_e3m2(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_NEONFHM
    nk_dots_pack_e3m2_neonfhm(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e3m2_genoa(b, n, k, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e3m2_skylake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e3m2_haswell(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_e3m2_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_NEONFHM
    nk_dots_packed_e3m2_neonfhm(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e3m2_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e3m2_skylake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e3m2_haswell(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_e3m2_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u4(nk_size_t n, nk_size_t k) {
#if NK_TARGET_ICELAKE
    return nk_dots_packed_size_u4_icelake(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_u4_neonsdot(n, k);
#else
    return nk_dots_packed_size_u4_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_u4(nk_u4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_ICELAKE
    nk_dots_pack_u4_icelake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_u4_neonsdot(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_u4_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                 nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_ICELAKE
    nk_dots_packed_u4_icelake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_u4_neonsdot(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_u4_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_i4(nk_size_t n, nk_size_t k) {
#if NK_TARGET_ICELAKE
    return nk_dots_packed_size_i4_icelake(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_i4_neonsdot(n, k);
#else
    return nk_dots_packed_size_i4_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_pack_i4(nk_i4x2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_ICELAKE
    nk_dots_pack_i4_icelake(b, n, k, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_i4_neonsdot(b, n, k, b_stride, b_packed);
#else
    nk_dots_pack_i4_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                 nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_ICELAKE
    nk_dots_packed_i4_icelake(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_i4_neonsdot(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_packed_i4_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f16(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_f16_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONHALF
    nk_dots_symmetric_f16_neonhalf(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_dots_symmetric_f16_neonfhm(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f16_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f16_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_f16_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_bf16_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_bf16_sapphireamx(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONBFDOT
    nk_dots_symmetric_bf16_neonbfdot(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_bf16_genoa(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_bf16_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_bf16_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_bf16_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_i8(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                    nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_i8_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_i8_sapphireamx(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_i8_neonsdot(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_i8_icelake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SIERRA
    nk_dots_symmetric_i8_sierra(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_i8_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_i8_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_u8(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                    nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_u8_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_u8_icelake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SIERRA
    nk_dots_symmetric_u8_sierra(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_u8_neonsdot(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_u8_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_u8_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e4m3_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e4m3_genoa(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e4m3_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e4m3_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_e4m3_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e5m2_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e5m2_genoa(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e5m2_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e5m2_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_e5m2_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_NEONFHM
    nk_dots_symmetric_e2m3_neonfhm(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e2m3_genoa(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e2m3_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e2m3_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_e2m3_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                      nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_NEONFHM
    nk_dots_symmetric_e3m2_neonfhm(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e3m2_genoa(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e3m2_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e3m2_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_e3m2_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                    nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_u4_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_u4_icelake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_u4_neonsdot(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_u4_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                    nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_i4_sme(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_i4_icelake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_i4_neonsdot(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_i4_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f32(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_dots_symmetric_f32_smef64(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f32_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f32_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_f32_neon(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_f32_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                     nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_dots_symmetric_f64_smef64(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f64_skylake(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f64_haswell(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_f64_neon(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_f64_serial(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
