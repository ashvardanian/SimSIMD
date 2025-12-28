/**
 *  @brief SIMD-accelerated 1-to-N dot product kernels for similarity and distance.
 *  @file include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *
 *  Implements batch dot-product kernels computing C[m×n] = A[m×k] × B[n×k]ᵀ
 *  with row-major inputs, optimized for ML inference and similarity workloads.
 *
 *  Primary Use Cases (1-to-N focus):
 *  - k-NN search: ||a-b||² = ||a||² + ||b||² - 2(a·b)
 *  - Cosine similarity: (a·b) / (||a||·||b||)
 *  - Sparse attention patterns
 *  - Embedding similarity matrices
 *  - k-means clustering, DBSCAN, hierarchical clustering
 *
 *  For datatypes (MKL-style naming: input×input→output):
 *  - bf16bf16f32: BF16 inputs accumulating to F32
 *  - i8i8i32: INT8 inputs accumulating to INT32
 *  - f32f32f32: F32 inputs accumulating to F32
 *  - bf16bf16bf16: BF16 with compact output
 *  - i8i8i8: INT8 with renormalized output
 *
 *  For hardware architectures:
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
 *  // Standard matmul: C[m×n] = A[m×k] × B[k×n]
 *  // B is stored row-major as k rows of n elements
 *  // Treat it as Bᵀ: n rows of k elements with stride = sizeof(element)
 *  nk_dots_bf16bf16f32_pack(b, n, k, sizeof(nk_bf16_t), b_packed);
 *  nk_dots_bf16bf16f32(a, b_packed, c, m, n, k, a_stride, c_stride);
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
 *  // Similarity search: C[m×n] = queries[m×k] × database[n×k]ᵀ
 *  // Both matrices stored row-major, each row is one vector of dimension k
 *  nk_size_t packed_bytes = nk_dots_bf16bf16f32_packed_size(n, k);
 *  void *b_packed = malloc(packed_bytes);
 *  nk_dots_bf16bf16f32_pack(database, n, k, k * sizeof(nk_bf16_t), b_packed);
 *  nk_dots_bf16bf16f32(queries, b_packed, c, m, n, k, ...);
 *  // Result: C[i,j] = dot(query i, database vector j)
 *  @endcode
 *
 *  The packed format is opaque and backend-specific. AMX expects (16×32) tiles with interleaved
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
 *  BLAS-style `C = α·A·B + β·C` scaling was considered but omitted. While useful for scientific
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
#include "numkong/dot.h" // nk_bf16x16_to_f32x16_skylake_

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for BF16 B matrix.
 *
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_dots_bf16bf16f32_packed_size(nk_size_t n, nk_size_t k);

/**
 *  @brief Packs BF16 B matrix into a backend-specific layout.
 *
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from nk_dots_bf16bf16f32_packed_size.
 */
NK_DYNAMIC void nk_dots_bf16bf16f32_pack( //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed BF16 B, accumulating into F32.
 *
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced by nk_dots_bf16bf16f32_pack.
 *  @param[out] c The output C matrix in row-major order (F32).
 *  @param[in] m The number of rows in A.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
NK_DYNAMIC void nk_dots_bf16bf16f32( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k, nk_size_t a_stride,
    nk_size_t c_stride);

/**
 *  @brief Compacts an F32 C matrix into BF16 in-place.
 *
 *  @param[in,out] c The output matrix buffer; contains F32 input and BF16 output.
 *  @param[in] m The number of rows in C.
 *  @param[in] n The number of columns in C.
 *  @param[in] c_stride The row stride in bytes for the F32 input.
 */
NK_DYNAMIC void nk_dots_bf16bf16bf16( //
    void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);

/**
 *  @brief Returns packed buffer size in bytes for I8 B matrix.
 *
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 */
NK_DYNAMIC nk_size_t nk_dots_i8i8i32_packed_size(nk_size_t n, nk_size_t k);

/**
 *  @brief Packs I8 B matrix into a backend-specific layout.
 *
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from nk_dots_i8i8i32_packed_size.
 */
NK_DYNAMIC void nk_dots_i8i8i32_pack( //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed I8 B, accumulating into I32.
 *
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced by nk_dots_i8i8i32_pack.
 *  @param[out] c The output C matrix in row-major order (I32).
 *  @param[in] m The number of rows in A.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
NK_DYNAMIC void nk_dots_i8i8i32( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n, nk_size_t k, nk_size_t a_stride,
    nk_size_t c_stride);

/**
 *  @brief Compacts an I32 C matrix into I8 using precomputed squared norms.
 *
 *  @param[in,out] c The output matrix buffer; contains I32 input and I8 output.
 *  @param[in] m The number of rows in C.
 *  @param[in] n The number of columns in C.
 *  @param[in] c_stride The row stride in bytes for the I32 input.
 *  @param[in] a_squared_norms Row norms for A (length m).
 *  @param[in] b_squared_norms Row norms for B (length n).
 */
NK_DYNAMIC void nk_dots_i8i8i8( //
    void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride, nk_i32_t const *a_squared_norms,
    nk_i32_t const *b_squared_norms);

/** @copydoc nk_dots_bf16bf16f32_packed_size */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_bf16bf16f32_pack */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_serial(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                               void *b_packed);
/** @copydoc nk_dots_bf16bf16f32 */
NK_PUBLIC void nk_dots_bf16bf16f32_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                          nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_bf16bf16bf16 */
NK_PUBLIC void nk_dots_bf16bf16bf16_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);

/** @copydoc nk_dots_i8i8i32_packed_size */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_i8i8i32_pack */
NK_PUBLIC void nk_dots_i8i8i32_pack_serial(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_i8i8i32 */
NK_PUBLIC void nk_dots_i8i8i32_serial(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_i8i8i8 */
NK_PUBLIC void nk_dots_i8i8i8_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                     nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products and VPDPBUSD for INT8.
 *  Packing interleaves elements for efficient SIMD broadcast patterns.
 */
#if NK_TARGET_GENOA
/** @copydoc nk_dots_i8i8i32_packed_size */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_i8i8i32_pack */
NK_PUBLIC void nk_dots_i8i8i32_pack_genoa(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                          void *b_packed);
/** @copydoc nk_dots_i8i8i32 */
NK_PUBLIC void nk_dots_i8i8i32_genoa(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_i8i8i8 */
NK_PUBLIC void nk_dots_i8i8i8_genoa(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                    nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);

/** @copydoc nk_dots_bf16bf16f32_packed_size */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_bf16bf16f32_pack */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_genoa(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                              void *b_packed);
/** @copydoc nk_dots_bf16bf16f32 */
NK_PUBLIC void nk_dots_bf16bf16f32_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                         nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_bf16bf16bf16 */
NK_PUBLIC void nk_dots_bf16bf16bf16_genoa(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);
#endif // NK_TARGET_GENOA

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16×32) BF16 or (16×64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
#if NK_TARGET_SAPPHIRE_AMX
/** @copydoc nk_dots_bf16bf16f32_packed_size */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_bf16bf16f32_pack */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_sapphire_amx(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                                     void *b_packed);
/** @copydoc nk_dots_bf16bf16f32 */
NK_PUBLIC void nk_dots_bf16bf16f32_sapphire_amx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m,
                                                nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_bf16bf16bf16 */
NK_PUBLIC void nk_dots_bf16bf16bf16_sapphire_amx(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride);

/** @copydoc nk_dots_i8i8i32_packed_size */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k);
/** @copydoc nk_dots_i8i8i32_pack */
NK_PUBLIC void nk_dots_i8i8i32_pack_sapphire_amx(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                                 void *b_packed);
/** @copydoc nk_dots_i8i8i32 */
NK_PUBLIC void nk_dots_i8i8i32_sapphire_amx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m,
                                            nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_i8i8i8 */
NK_PUBLIC void nk_dots_i8i8i8_sapphire_amx(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                           nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms);
#endif // NK_TARGET_SAPPHIRE_AMX

#include "numkong/dots/serial.h"
#include "numkong/dots/haswell.h"
#include "numkong/dots/skylake.h"
#include "numkong/dots/genoa.h"
#include "numkong/dots/sapphire.h"
#include "numkong/dots/sapphire_amx.h"
#include "numkong/dots/neon.h"
#include "numkong/dots/neonsdot.h"
#include "numkong/dots/neonhalf.h"
#include "numkong/dots/neonbfdot.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SAPPHIRE_AMX
    return nk_dots_bf16bf16f32_packed_size_sapphire_amx(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_bf16bf16f32_packed_size_genoa(n, k);
#else
    return nk_dots_bf16bf16f32_packed_size_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_bf16bf16f32_pack(nk_bf16_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride,
                                        void *b_packed) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_bf16bf16f32_pack_sapphire_amx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_bf16bf16f32_pack_genoa(b, n, k, b_stride, b_packed);
#else
    nk_dots_bf16bf16f32_pack_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_bf16bf16f32(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                   nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_bf16bf16f32_sapphire_amx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_bf16bf16f32_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_bf16bf16f32_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_bf16bf16bf16(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_bf16bf16bf16_sapphire_amx(c, m, n, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_bf16bf16bf16_genoa(c, m, n, c_stride);
#else
    nk_dots_bf16bf16bf16_serial(c, m, n, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SAPPHIRE_AMX
    return nk_dots_i8i8i32_packed_size_sapphire_amx(n, k);
#elif NK_TARGET_GENOA
    return nk_dots_i8i8i32_packed_size_genoa(n, k);
#else
    return nk_dots_i8i8i32_packed_size_serial(n, k);
#endif
}

NK_PUBLIC void nk_dots_i8i8i32_pack(nk_i8_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_i8i8i32_pack_sapphire_amx(b, n, k, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_i8i8i32_pack_genoa(b, n, k, b_stride, b_packed);
#else
    nk_dots_i8i8i32_pack_serial(b, n, k, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_i8i8i32(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                               nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_i8i8i32_sapphire_amx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_i8i8i32_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    nk_dots_i8i8i32_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_i8i8i8(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride, nk_i32_t const *a_squared_norms,
                              nk_i32_t const *b_squared_norms) {
#if NK_TARGET_SAPPHIRE_AMX
    nk_dots_i8i8i8_sapphire_amx(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#elif NK_TARGET_GENOA
    nk_dots_i8i8i8_genoa(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#else
    nk_dots_i8i8i8_serial(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
