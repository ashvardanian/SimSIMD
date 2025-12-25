/**
 *  @brief SIMD-accelerated 1-to-N dot product kernels for similarity and distance.
 *  @file include/simsimd/dots.h
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
 *  Performance Notes:
 *  - Targets MKL-competitive performance for supported type combinations
 *  - Outer-product algorithm optimized for 1-to-N (single query vs database)
 *  - Uses vdpbf16ps (BF16) and vpdpwssd (I8 VNNI) for maximum throughput
 *  - Best configs: BF16 NR=32/K_TILE=8/MR=4, I8 NR=64/K_TILE=8/MR=2
 *
 *  Hardware Throughput (Sapphire Rapids measured):
 *  - vdpbf16ps: ~0.7 instr/cycle → ~95 GFLOPS peak (64 FLOPs/instr)
 *  - vfmadd512: ~2.5 instr/cycle → ~168 GFLOPS peak (32 FLOPs/instr)
 *  - vpdpwssd: Higher throughput than BF16, exceeds sgemm for I8
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
 *  simsimd_dots_bf16bf16f32_pack(b, n, k, sizeof(simsimd_bf16_t), b_packed);
 *  simsimd_dots_bf16bf16f32(a, b_packed, c, m, n, k, a_stride, c_stride);
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
 *  simsimd_size_t packed_bytes = simsimd_dots_bf16bf16f32_packed_size(n, k);
 *  void *b_packed = malloc(packed_bytes);
 *  simsimd_dots_bf16bf16f32_pack(database, n, k, k * sizeof(simsimd_bf16_t), b_packed);
 *  simsimd_dots_bf16bf16f32(queries, b_packed, c, m, n, k, ...);
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
#ifndef SIMSIMD_DOTS_H
#define SIMSIMD_DOTS_H

#include "types.h"

#include "dot.h" // `_simsimd_bf16x16_to_f32x16_skylake`

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
SIMSIMD_DYNAMIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size(simsimd_size_t n, simsimd_size_t k);

/**
 *  @brief Packs BF16 B matrix into a backend-specific layout.
 *
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from simsimd_dots_bf16bf16f32_packed_size.
 */
SIMSIMD_DYNAMIC void simsimd_dots_bf16bf16f32_pack( //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, simsimd_size_t b_stride, void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed BF16 B, accumulating into F32.
 *
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced by simsimd_dots_bf16bf16f32_pack.
 *  @param[out] c The output C matrix in row-major order (F32).
 *  @param[in] m The number of rows in A.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
SIMSIMD_DYNAMIC void simsimd_dots_bf16bf16f32( //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, simsimd_size_t m, simsimd_size_t n,
    simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

/**
 *  @brief Compacts an F32 C matrix into BF16 in-place.
 *
 *  @param[in,out] c The output matrix buffer; contains F32 input and BF16 output.
 *  @param[in] m The number of rows in C.
 *  @param[in] n The number of columns in C.
 *  @param[in] c_stride The row stride in bytes for the F32 input.
 */
SIMSIMD_DYNAMIC void simsimd_dots_bf16bf16bf16( //
    void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride);

/**
 *  @brief Returns packed buffer size in bytes for I8 B matrix.
 *
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 */
SIMSIMD_DYNAMIC simsimd_size_t simsimd_dots_i8i8i32_packed_size(simsimd_size_t n, simsimd_size_t k);

/**
 *  @brief Packs I8 B matrix into a backend-specific layout.
 *
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from simsimd_dots_i8i8i32_packed_size.
 */
SIMSIMD_DYNAMIC void simsimd_dots_i8i8i32_pack( //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, simsimd_size_t b_stride, void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed I8 B, accumulating into I32.
 *
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced by simsimd_dots_i8i8i32_pack.
 *  @param[out] c The output C matrix in row-major order (I32).
 *  @param[in] m The number of rows in A.
 *  @param[in] n The number of rows in B (output columns).
 *  @param[in] k The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
SIMSIMD_DYNAMIC void simsimd_dots_i8i8i32( //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
    simsimd_size_t a_stride, simsimd_size_t c_stride);

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
SIMSIMD_DYNAMIC void simsimd_dots_i8i8i8( //
    void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride, simsimd_i32_t const *a_squared_norms,
    simsimd_i32_t const *b_squared_norms);

/*  Tunable tile sizes for cache blocking.
 *  These can be overridden before including this header to tune for specific cache sizes.
 *  The L1 tile should fit 3 matrices (A, B, C) in L1 cache with room for accumulators.
 */
#ifndef SIMSIMD_DOTS_L1_TILE_M
#define SIMSIMD_DOTS_L1_TILE_M 128
#endif
#ifndef SIMSIMD_DOTS_L1_TILE_N
#define SIMSIMD_DOTS_L1_TILE_N 128
#endif
#ifndef SIMSIMD_DOTS_L1_TILE_K
#define SIMSIMD_DOTS_L1_TILE_K 128
#endif

/*  Serial backends for packing and multiplication.
 *  These are portable reference implementations with no SIMD dependencies.
 *  Serial packing simply copies B transposed - no special layout required.
 */
/** @copydoc simsimd_dots_bf16bf16f32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size_serial(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_bf16bf16f32_pack */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_pack_serial(simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                         simsimd_size_t b_stride, void *b_packed);
/** @copydoc simsimd_dots_bf16bf16f32 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_serial(simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c,
                                                    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                                    simsimd_size_t a_stride, simsimd_size_t c_stride);
/** @copydoc simsimd_dots_bf16bf16bf16 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_serial(void *c, simsimd_size_t m, simsimd_size_t n,
                                                     simsimd_size_t c_stride);

/** @copydoc simsimd_dots_i8i8i32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_i8i8i32_packed_size_serial(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_i8i8i32_pack */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_pack_serial(simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                     simsimd_size_t b_stride, void *b_packed);
/** @copydoc simsimd_dots_i8i8i32 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_serial(simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c,
                                                simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                                simsimd_size_t a_stride, simsimd_size_t c_stride);
/** @copydoc simsimd_dots_i8i8i8 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i8_serial(void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride,
                                               simsimd_i32_t const *a_squared_norms,
                                               simsimd_i32_t const *b_squared_norms);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products and VPDPBUSD for INT8.
 *  Packing interleaves elements for efficient SIMD broadcast patterns.
 */
#if SIMSIMD_TARGET_GENOA
/** @copydoc simsimd_dots_i8i8i32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_i8i8i32_packed_size_genoa(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_i8i8i32_pack */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_pack_genoa(simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                    simsimd_size_t b_stride, void *b_packed);
/** @copydoc simsimd_dots_i8i8i32 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_genoa(simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c,
                                               simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                               simsimd_size_t a_stride, simsimd_size_t c_stride);
/** @copydoc simsimd_dots_i8i8i8 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i8_genoa(void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride,
                                              simsimd_i32_t const *a_squared_norms,
                                              simsimd_i32_t const *b_squared_norms);

/** @copydoc simsimd_dots_bf16bf16f32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size_genoa(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_bf16bf16f32_pack */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_pack_genoa(simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                        simsimd_size_t b_stride, void *b_packed);
/** @copydoc simsimd_dots_bf16bf16f32 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_genoa(simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c,
                                                   simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                                   simsimd_size_t a_stride, simsimd_size_t c_stride);
/** @copydoc simsimd_dots_bf16bf16bf16 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_genoa(void *c, simsimd_size_t m, simsimd_size_t n,
                                                    simsimd_size_t c_stride);
#endif // SIMSIMD_TARGET_GENOA

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16×32) BF16 or (16×64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
#if SIMSIMD_TARGET_SAPPHIRE_AMX
/** @copydoc simsimd_dots_bf16bf16f32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_bf16bf16f32_pack */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_pack_sapphire_amx(simsimd_bf16_t const *b, simsimd_size_t n,
                                                               simsimd_size_t k, simsimd_size_t b_stride,
                                                               void *b_packed);
/** @copydoc simsimd_dots_bf16bf16f32 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_sapphire_amx(simsimd_bf16_t const *a, void const *b_packed,
                                                          simsimd_f32_t *c, simsimd_size_t m, simsimd_size_t n,
                                                          simsimd_size_t k, simsimd_size_t a_stride,
                                                          simsimd_size_t c_stride);
/** @copydoc simsimd_dots_bf16bf16bf16 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_sapphire_amx(void *c, simsimd_size_t m, simsimd_size_t n,
                                                           simsimd_size_t c_stride);

/** @copydoc simsimd_dots_i8i8i32_packed_size */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_i8i8i32_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k);
/** @copydoc simsimd_dots_i8i8i32_pack */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_pack_sapphire_amx(simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                           simsimd_size_t b_stride, void *b_packed);
/** @copydoc simsimd_dots_i8i8i32 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_sapphire_amx(simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c,
                                                      simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                                      simsimd_size_t a_stride, simsimd_size_t c_stride);
/** @copydoc simsimd_dots_i8i8i8 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i8_sapphire_amx(void *c, simsimd_size_t m, simsimd_size_t n,
                                                     simsimd_size_t c_stride, simsimd_i32_t const *a_squared_norms,
                                                     simsimd_i32_t const *b_squared_norms);
#endif // SIMSIMD_TARGET_SAPPHIRE_AMX (declarations)

/*  Legacy unpacked matmul implementations for direct A×B^T operations.
 *  These operate on unpacked matrices directly (no two-phase pack/multiply API).
 *  Use when B matrix is not reused across multiple multiplications.
 *
 *  Naming: simsimd_dots_<type>_<variant>_unpacked
 *
 *  - serial: basic tiled implementation
 *  - accurate: higher precision accumulator
 *
 */
#define SIMSIMD_MAKE_DOTS_UNPACKED(name, input_type, accumulator_type, output_type, load_and_convert,                \
                                   convert_and_store)                                                                \
    SIMSIMD_PUBLIC void simsimd_dots_##input_type##_##name##_unpacked(                                               \
        simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols, simsimd_##input_type##_t const *a,        \
        simsimd_size_t a_stride, simsimd_##input_type##_t const *b, simsimd_size_t b_stride,                         \
        simsimd_##output_type##_t *c, simsimd_size_t c_stride) {                                                     \
        for (simsimd_size_t i = 0; i < a_rows; ++i) {                                                                \
            simsimd_##input_type##_t const *a_row = (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes(     \
                (void *)a, i * a_stride);                                                                            \
            simsimd_##output_type##_t *c_row = (simsimd_##output_type##_t *)_simsimd_advance_by_bytes((void *)c,     \
                                                                                                      i * c_stride); \
            for (simsimd_size_t j = 0; j < b_rows; ++j) {                                                            \
                simsimd_##input_type##_t const *b_row = (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes( \
                    (void *)b, j * b_stride);                                                                        \
                simsimd_##accumulator_type##_t sum = 0;                                                              \
                for (simsimd_size_t k = 0; k < cols; ++k) {                                                          \
                    simsimd_##accumulator_type##_t aik = load_and_convert(a_row + k);                                \
                    simsimd_##accumulator_type##_t bjk = load_and_convert(b_row + k);                                \
                    sum += aik * bjk;                                                                                \
                }                                                                                                    \
                convert_and_store(sum, c_row + j);                                                                   \
            }                                                                                                        \
        }                                                                                                            \
    }

#define SIMSIMD_MAKE_TILED_UNPACKED(name, input_type, accumulator_type, output_type, load_and_convert,                \
                                    convert_and_store, tile_size)                                                     \
    SIMSIMD_PUBLIC void simsimd_dots_##input_type##_##name##_unpacked(                                                \
        simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols, simsimd_##input_type##_t const *a,         \
        simsimd_size_t a_stride, simsimd_##input_type##_t const *b, simsimd_size_t b_stride,                          \
        simsimd_##output_type##_t *c, simsimd_size_t c_stride) {                                                      \
        for (simsimd_size_t ii = 0; ii < a_rows; ii += tile_size) {                                                   \
            for (simsimd_size_t jj = 0; jj < b_rows; jj += tile_size) {                                               \
                for (simsimd_size_t kk = 0; kk < cols; kk += tile_size) {                                             \
                    simsimd_size_t i_max = (ii + tile_size < a_rows) ? (ii + tile_size) : a_rows;                     \
                    simsimd_size_t j_max = (jj + tile_size < b_rows) ? (jj + tile_size) : b_rows;                     \
                    simsimd_size_t k_max = (kk + tile_size < cols) ? (kk + tile_size) : cols;                         \
                    for (simsimd_size_t i = ii; i < i_max; ++i) {                                                     \
                        simsimd_##input_type##_t const *a_row =                                                       \
                            (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)a, i * a_stride);     \
                        simsimd_##output_type##_t *c_row = (simsimd_##output_type##_t *)_simsimd_advance_by_bytes(    \
                            (void *)c, i * c_stride);                                                                 \
                        for (simsimd_size_t j = jj; j < j_max; ++j) {                                                 \
                            simsimd_##input_type##_t const *b_row =                                                   \
                                (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)b, j * b_stride); \
                            simsimd_##accumulator_type##_t sum = 0;                                                   \
                            for (simsimd_size_t k = kk; k < k_max; ++k) {                                             \
                                simsimd_##accumulator_type##_t aik = load_and_convert(a_row + k);                     \
                                simsimd_##accumulator_type##_t bjk = load_and_convert(b_row + k);                     \
                                sum += aik * bjk;                                                                     \
                            }                                                                                         \
                            convert_and_store(sum, c_row + j);                                                        \
                        }                                                                                             \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

SIMSIMD_MAKE_TILED_UNPACKED(serial, f64, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT,
                            16) // simsimd_dots_f64f64f64_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, f32, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT,
                            16) // simsimd_dots_f32_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, f16, f32, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16,
                            16) // simsimd_dots_f16f16f32_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, bf16, f32, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16,
                            16) // simsimd_dots_bf16bf16f32_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, i8, i64, i8, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT,
                            16) // simsimd_dots_i8_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, f32, f64, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT,
                            16) // simsimd_dots_f32_accurate_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, f16, f64, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16,
                            16) // simsimd_dots_f16f16f32_accurate_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, bf16, f64, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16,
                            16) // simsimd_dots_bf16bf16f32_accurate_unpacked

/*  Cache-Blocked GEMM with Outer-Product Micro-Kernel
 *  ===================================================
 *  Uses 5-loop GotoBLAS structure with 3-level cache blocking:
 *    - L3 blocking: column_block_size (NC) columns at a time
 *    - L1 blocking: depth_block_size (KC) k-elements at a time
 *    - L2 blocking: row_block_size (MC) rows at a time
 *
 *  The innermost kernel uses the unified outer-product API which:
 *    - Processes k_tile elements per update (2 for f32, 4 for bf16/f16, 8 for i8)
 *    - Outputs 1×16 partial products (MR=1, NR=16)
 *    - Avoids horizontal reduction until finalize
 *
 *  Variable naming convention (descriptive snake_case):
 *    - row_*, col_*, depth_*: dimension indices
 *    - *_block_start: start of cache block
 *    - *_block_end: end of cache block
 *    - *_tile_start: start of register tile
 */

/**
 *  @brief Unified GEMM macro with MR×NR micro-kernel and cache blocking.
 *
 *  Processes MR rows together, amortizing B loads across rows. Uses 5-loop
 *  GotoBLAS structure with K-first order for optimal cache utilization.
 *
 *  The nr_acc parameter controls the number of parallel accumulator chains per row.
 *  This breaks the dependency chain in the K-loop, significantly improving throughput:
 *    - nr_acc=1: NR=16, 1 accumulator (baseline)
 *    - nr_acc=2: NR=32, 2 accumulators (2× throughput)
 *    - nr_acc=4: NR=64, 4 accumulators (4× throughput)
 *
 *  @param suffix         Function name suffix (e.g., f32f32f32_skylake)
 *  @param input_type     Input element type (f32, bf16, i8)
 *  @param output_type    Output element type (f32, i32)
 *  @param state_type     Accumulator state type (single 1×16 state)
 *  @param init_fn        State initialization function
 *  @param update_fn      State update function (processes k_tile elements)
 *  @param finalize_fn    State finalization function (outputs 16 elements)
 *  @param k_tile         K-dimension tile size per update (2 for f32x2, 4 for bf16x4, 8 for i8x8)
 *  @param mr_size        Number of rows to process together (register blocking M)
 *  @param nr_acc         Number of parallel accumulators per row (1, 2, or 4)
 *  @param mc_size        M-dimension cache block size (L2 blocking)
 *  @param nc_size        N-dimension cache block size (L3 blocking)
 *  @param kc_size        K-dimension cache block size (L1 blocking)
 */
#define SIMSIMD_MAKE_DOTS(suffix, input_type, output_type, state_type, init_fn, update_fn, finalize_fn, k_tile,       \
                          mr_size, nr_acc, mc_size, nc_size, kc_size)                                                 \
                                                                                                                      \
    /* GEMM: C[m×n] = A[m×k] × B^T[n×k] with B pre-packed for k-major access. */                                      \
    SIMSIMD_PUBLIC void simsimd_dots_##suffix(simsimd_##input_type##_t const *a_matrix, void const *b_packed_void,    \
                                              simsimd_##output_type##_t *c_matrix, simsimd_size_t row_count,          \
                                              simsimd_size_t column_count, simsimd_size_t depth,                      \
                                              simsimd_size_t a_stride, simsimd_size_t c_stride) {                     \
        simsimd_##input_type##_t const *b_packed = (simsimd_##input_type##_t const *)b_packed_void;                   \
                                                                                                                      \
        simsimd_size_t const nr_base = 16;          /* Columns per single accumulator */                              \
        simsimd_size_t const nr_size = 16 * nr_acc; /* Total columns per NR tile (16 × nr_acc) */                     \
                                                                                                                      \
        /* Loop 1: L3 cache blocking over columns (NC columns at a time) */                                           \
        for (simsimd_size_t nc_start = 0; nc_start < column_count; nc_start += nc_size) {                             \
            simsimd_size_t nc_end = nc_start + nc_size;                                                               \
            if (nc_end > column_count) nc_end = column_count;                                                         \
                                                                                                                      \
            /* Loop 2: L1 cache blocking over depth (KC elements at a time) */                                        \
            for (simsimd_size_t kc_start = 0; kc_start < depth; kc_start += kc_size) {                                \
                simsimd_size_t kc_end = kc_start + kc_size;                                                           \
                if (kc_end > depth) kc_end = depth;                                                                   \
                simsimd_size_t kc_len = kc_end - kc_start;                                                            \
                                                                                                                      \
                /* Loop 3: L2 cache blocking over rows (MC rows at a time) */                                         \
                for (simsimd_size_t mc_start = 0; mc_start < row_count; mc_start += mc_size) {                        \
                    simsimd_size_t mc_end = mc_start + mc_size;                                                       \
                    if (mc_end > row_count) mc_end = row_count;                                                       \
                                                                                                                      \
                    /* Loop 4: Register tiling over columns (NR = 16 * nr_acc columns per tile) */                    \
                    for (simsimd_size_t nr_start = nc_start; nr_start < nc_end; nr_start += nr_size) {                \
                        simsimd_size_t columns_in_tile = nr_size;                                                     \
                        if (nr_start + columns_in_tile > nc_end) columns_in_tile = nc_end - nr_start;                 \
                                                                                                                      \
                        /* Loop 5: Register tiling over rows (MR rows per tile) */                                    \
                        for (simsimd_size_t mr_start = mc_start; mr_start < mc_end; mr_start += mr_size) {            \
                            simsimd_size_t rows_in_tile = mr_size;                                                    \
                            if (mr_start + rows_in_tile > mc_end) rows_in_tile = mc_end - mr_start;                   \
                                                                                                                      \
                            /* Initialize MR × nr_acc accumulators (2D array for parallel chains) */                  \
                            state_type states[mr_size][nr_acc];                                                       \
                            for (simsimd_size_t r = 0; r < rows_in_tile; ++r) {                                       \
                                for (simsimd_size_t a = 0; a < (simsimd_size_t)nr_acc; ++a) {                         \
                                    init_fn(&states[r][a]);                                                           \
                                }                                                                                     \
                            }                                                                                         \
                                                                                                                      \
                            /* Loop 6: Accumulate over depth block (k_tile elements per iteration) */                 \
                            for (simsimd_size_t k_offset = 0; k_offset + k_tile <= kc_len; k_offset += k_tile) {      \
                                /* Update all MR rows with all nr_acc accumulator chains */                           \
                                for (simsimd_size_t r = 0; r < rows_in_tile; ++r) {                                   \
                                    simsimd_##input_type##_t const *a_ptr =                                           \
                                        (simsimd_##input_type##_t const *)((char const *)a_matrix +                   \
                                                                           (mr_start + r) * a_stride) +               \
                                        kc_start + k_offset;                                                          \
                                    /* Process nr_acc independent columns (breaks dependency chain) */                \
                                    for (simsimd_size_t a = 0; a < (simsimd_size_t)nr_acc; ++a) {                     \
                                        simsimd_size_t nr_offset = nr_start + a * nr_base;                            \
                                        if (nr_offset >= nc_end) break; /* Skip if beyond edge */                     \
                                        simsimd_##input_type##_t const *b_ptr =                                       \
                                            b_packed + (nr_offset / nr_base) * depth * nr_base + kc_start * nr_base + \
                                            k_offset * nr_base;                                                       \
                                        update_fn(&states[r][a], a_ptr, b_ptr);                                       \
                                    }                                                                                 \
                                }                                                                                     \
                            }                                                                                         \
                                                                                                                      \
                            /* Finalize and store MR × nr_acc × 16 result tiles */                                    \
                            for (simsimd_size_t r = 0; r < rows_in_tile; ++r) {                                       \
                                for (simsimd_size_t a = 0; a < (simsimd_size_t)nr_acc; ++a) {                         \
                                    simsimd_size_t nr_offset = nr_start + a * nr_base;                                \
                                    if (nr_offset >= nc_end) break;                                                   \
                                    simsimd_size_t cols = nr_base;                                                    \
                                    if (nr_offset + cols > nc_end) cols = nc_end - nr_offset;                         \
                                                                                                                      \
                                    simsimd_##output_type##_t result_tile[16];                                        \
                                    finalize_fn(&states[r][a], result_tile);                                          \
                                                                                                                      \
                                    simsimd_##output_type##_t *c_row_ptr =                                            \
                                        (simsimd_##output_type##_t *)((char *)c_matrix + (mr_start + r) * c_stride) + \
                                        nr_offset;                                                                    \
                                    for (simsimd_size_t col = 0; col < cols; ++col) {                                 \
                                        c_row_ptr[col] += result_tile[col];                                           \
                                    }                                                                                 \
                                }                                                                                     \
                            }                                                                                         \
                        }                                                                                             \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

// Convenience macro with nr_acc=1 (single accumulator chain)
#define SIMSIMD_MAKE_DOTS_MRx16(suffix, input_type, output_type, state_type, init_fn, update_fn, finalize_fn, k_tile, \
                                mr_size, mc_size, nc_size, kc_size)                                                   \
    SIMSIMD_MAKE_DOTS(suffix, input_type, output_type, state_type, init_fn, update_fn, finalize_fn, k_tile, mr_size,  \
                      /*nr_acc=*/1, mc_size, nc_size, kc_size)

/*  Packed buffer header for tiled layout (64-byte aligned).
 *  Used by all packed matmul backends (serial, AVX-512, AMX, SVE).
 */
typedef struct {
    simsimd_u32_t full_n_tiles;  // Number of full N tiles (TILE_N rows each)
    simsimd_u32_t full_k_tiles;  // Number of K tiles (TILE_K cols each, includes remainder)
    simsimd_u16_t n_edge_rows;   // Remaining N rows (for edge handling)
    simsimd_u16_t n_edge_offset; // Offset to N edge region (for AMX hybrid layout)
    simsimd_u32_t reserved[12];  // Padding to 64 bytes
} simsimd_dots_packed_header_t;

/*  Tiled row-major packed format for cache-efficient matrix multiplication.
 *
 *  Layout: B matrix is divided into tiles of TILE_N rows × TILE_K columns.
 *  Tiles are stored in row-major order: k-tiles vary slowest, n-tiles vary fastest.
 *  Within each tile, elements are stored row-major (n-row × k-elements).
 *
 *  Tile widths are 64 bytes (1 cache line) for optimal memory access:
 *    - f64: 8 elements per tile row
 *    - f32: 16 elements per tile row
 *    - f16/bf16: 32 elements per tile row
 *    - i8/u8: 64 elements per tile row
 *
 *  Tile height is 16 rows, matching streaming dot-product kernel batch sizes.
 *
 *  Memory layout for packed B (n=48, k=64, f32):
 *    n_tiles = ceil(48/16) = 3, k_tiles = ceil(64/16) = 4
 *    Tiles stored as: [kt=0,nt=0] [kt=0,nt=1] [kt=0,nt=2] [kt=1,nt=0] ...
 *    Each tile: 16 rows × 16 elements = 256 f32 = 1KB
 */

// Serial tile dimensions: 64-byte width (1 cache line), 16 rows height
#define SIMSIMD_DOTS_SERIAL_TILE_N      16
#define SIMSIMD_DOTS_SERIAL_TILE_K_F64  8  // 8 × 8 bytes = 64 bytes
#define SIMSIMD_DOTS_SERIAL_TILE_K_F32  16 // 16 × 4 bytes = 64 bytes
#define SIMSIMD_DOTS_SERIAL_TILE_K_F16  32 // 32 × 2 bytes = 64 bytes
#define SIMSIMD_DOTS_SERIAL_TILE_K_BF16 32 // 32 × 2 bytes = 64 bytes
#define SIMSIMD_DOTS_SERIAL_TILE_K_I8   64 // 64 × 1 byte = 64 bytes
#define SIMSIMD_DOTS_SERIAL_TILE_K_U8   64 // 64 × 1 byte = 64 bytes

// Helper to get tile_k for a given type (serial implementation)
#define SIMSIMD_DOTS_SERIAL_TILE_K(input_type)                                  \
    ((sizeof(simsimd_##input_type##_t) == 8)   ? SIMSIMD_DOTS_SERIAL_TILE_K_F64 \
     : (sizeof(simsimd_##input_type##_t) == 4) ? SIMSIMD_DOTS_SERIAL_TILE_K_F32 \
     : (sizeof(simsimd_##input_type##_t) == 2) ? SIMSIMD_DOTS_SERIAL_TILE_K_F16 \
                                               : SIMSIMD_DOTS_SERIAL_TILE_K_I8)

/**
 *  @brief Macro to generate packed_size function for serial tiled row-major format.
 *
 *  Calculates buffer size needed for packed B matrix including header and padding.
 *  Tiles are padded to full size even for edge cases to simplify access patterns.
 *  Uses MKL-style naming: simsimd_dots_{input}{input}{output}_packed_size_{suffix}
 */
#define SIMSIMD_MAKE_DOTS_SERIAL_PACKED_SIZE(suffix, input_type, output_type, tile_k)                        \
    SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_##input_type##input_type##output_type##_packed_size_##suffix( \
        simsimd_size_t n, simsimd_size_t k) {                                                                \
        simsimd_size_t const tile_n = SIMSIMD_DOTS_SERIAL_TILE_N;                                            \
        simsimd_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                            \
        simsimd_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                            \
        simsimd_size_t const tile_size = tile_n * tile_k * sizeof(simsimd_##input_type##_t);                 \
        return sizeof(simsimd_dots_packed_header_t) + n_tiles * k_tiles * tile_size;                         \
    }

/**
 *  @brief Macro to generate pack function for serial tiled row-major format.
 *
 *  Packs B matrix into tiles: k-tiles outer loop, n-tiles inner loop.
 *  Each tile contains TILE_N rows × TILE_K elements in row-major order.
 *  Edge tiles are zero-padded to full tile size.
 *  Uses MKL-style naming: simsimd_dots_{input}{input}{output}_pack_{suffix}
 */
#define SIMSIMD_MAKE_DOTS_SERIAL_PACK(suffix, input_type, output_type, tile_k)                                   \
    SIMSIMD_PUBLIC void simsimd_dots_##input_type##input_type##output_type##_pack_##suffix(                      \
        simsimd_##input_type##_t const *b, simsimd_size_t n, simsimd_size_t k, simsimd_size_t b_stride,          \
        void *b_packed) {                                                                                        \
                                                                                                                 \
        simsimd_size_t const tile_n = SIMSIMD_DOTS_SERIAL_TILE_N;                                                \
        simsimd_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                                \
        simsimd_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                                \
        simsimd_size_t const tile_size = tile_n * tile_k;                                                        \
                                                                                                                 \
        /* Store dimensions in header */                                                                         \
        simsimd_dots_packed_header_t *header = (simsimd_dots_packed_header_t *)b_packed;                         \
        header->full_n_tiles = (simsimd_u32_t)n_tiles;                                                           \
        header->full_k_tiles = (simsimd_u32_t)k_tiles;                                                           \
        header->n_edge_rows = (simsimd_u16_t)(n % tile_n);                                                       \
        header->n_edge_offset = (simsimd_u16_t)(n - (n % tile_n));                                               \
                                                                                                                 \
        simsimd_##input_type##_t *packed = (simsimd_##input_type##_t *)((char *)b_packed +                       \
                                                                        sizeof(simsimd_dots_packed_header_t));   \
                                                                                                                 \
        /* Zero entire buffer for edge tile padding */                                                           \
        for (simsimd_size_t i = 0; i < n_tiles * k_tiles * tile_size; ++i) packed[i] = 0;                        \
                                                                                                                 \
        /* Pack tiles: k-tiles outer, n-tiles inner */                                                           \
        for (simsimd_size_t kt = 0; kt < k_tiles; ++kt) {                                                        \
            simsimd_size_t const k_start = kt * tile_k;                                                          \
            simsimd_size_t const k_end = (k_start + tile_k < k) ? (k_start + tile_k) : k;                        \
                                                                                                                 \
            for (simsimd_size_t nt = 0; nt < n_tiles; ++nt) {                                                    \
                simsimd_size_t const n_start = nt * tile_n;                                                      \
                simsimd_size_t const n_end = (n_start + tile_n < n) ? (n_start + tile_n) : n;                    \
                                                                                                                 \
                simsimd_size_t const tile_idx = kt * n_tiles + nt;                                               \
                simsimd_##input_type##_t *tile = packed + tile_idx * tile_size;                                  \
                                                                                                                 \
                /* Copy B rows into tile (row-major within tile) */                                              \
                for (simsimd_size_t ni = n_start; ni < n_end; ++ni) {                                            \
                    simsimd_##input_type##_t const *b_row = (simsimd_##input_type##_t const *)((char const *)b + \
                                                                                               ni * b_stride);   \
                    simsimd_size_t const row_in_tile = ni - n_start;                                             \
                                                                                                                 \
                    for (simsimd_size_t ki = k_start; ki < k_end; ++ki) {                                        \
                        simsimd_size_t const col_in_tile = ki - k_start;                                         \
                        tile[row_in_tile * tile_k + col_in_tile] = b_row[ki];                                    \
                    }                                                                                            \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }

/**
 *  @brief Unified macro to generate packed buffer size for GEMM.
 *
 *  Layout: (n/16) column-blocks × k_padded × 16 elements.
 *  K is padded to the specified alignment for SIMD requirements.
 *  Uses naming: simsimd_dots_{input}{input}{output}_packed_size_{suffix}
 *
 *  @param suffix       Backend suffix (e.g., skylake, genoa, ice)
 *  @param input_type   Input scalar type (e.g., f32, bf16, i8)
 *  @param output_type  Output scalar type (e.g., f32, i32)
 *  @param k_align      K dimension alignment (1 for F32, 16 for BF16, 8 for I8)
 */
#define SIMSIMD_MAKE_DOTS_PACKED_SIZE(suffix, input_type, output_type, k_align)                              \
    SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_##input_type##input_type##output_type##_packed_size_##suffix( \
        simsimd_size_t n, simsimd_size_t k) {                                                                \
        simsimd_size_t const n_blocks = (n + 15) / 16;                                                       \
        simsimd_size_t const k_padded = (k + (k_align) - 1) & ~((simsimd_size_t)(k_align) - 1);              \
        return n_blocks * k_padded * 16 * sizeof(simsimd_##input_type##_t);                                  \
    }

/**
 *  @brief Unified macro to generate pack function for GEMM.
 *
 *  Packs B matrix (n×k row-major) into interleaved layout within 16-column tiles.
 *  The interleave factor determines how many consecutive K values are grouped:
 *    - F32 (interleave=1): dst[ki * 16 + col] = b[col, ki]
 *    - BF16 (interleave=2): dst[group * 32 + col * 2 + i] = b[col, group*2 + i]  (for VDPBF16PS)
 *    - I8 (interleave=4): dst[group * 64 + col * 4 + i] = b[col, group*4 + i]  (for VPDPBUSD)
 *
 *  @param suffix       Backend suffix (e.g., skylake, genoa, ice)
 *  @param input_type   Input scalar type (e.g., f32, bf16, i8)
 *  @param output_type  Output scalar type (e.g., f32, i32)
 *  @param interleave   Number of K values to group (1=none, 2=pairs, 4=quads)
 *  @param k_align      K dimension alignment for padding (1, 8, 16, etc.)
 */
#define SIMSIMD_MAKE_DOTS_PACK(suffix, input_type, output_type, interleave, k_align)                            \
    SIMSIMD_PUBLIC void simsimd_dots_##input_type##input_type##output_type##_pack_##suffix(                     \
        simsimd_##input_type##_t const *b, simsimd_size_t n, simsimd_size_t k, simsimd_size_t b_stride,         \
        void *b_packed) {                                                                                       \
        simsimd_##input_type##_t *bp = (simsimd_##input_type##_t *)b_packed;                                    \
        simsimd_size_t const stride_elements = b_stride / sizeof(simsimd_##input_type##_t);                     \
        simsimd_size_t const k_padded = (k + (k_align) - 1) & ~((simsimd_size_t)(k_align) - 1);                 \
        simsimd_size_t const num_groups = k_padded / (interleave);                                              \
        for (simsimd_size_t col_block = 0; col_block < n; col_block += 16) {                                    \
            simsimd_##input_type##_t *dst_block = bp + (col_block / 16) * k_padded * 16;                        \
            for (simsimd_size_t group = 0; group < num_groups; ++group) {                                       \
                for (simsimd_size_t col = 0; col < 16; ++col) {                                                 \
                    simsimd_size_t const src_col = col_block + col;                                             \
                    simsimd_##input_type##_t const *src = (src_col < n) ? b + src_col * stride_elements : NULL; \
                    for (simsimd_size_t i = 0; i < (interleave); ++i) {                                         \
                        simsimd_size_t const ki = group * (interleave) + i;                                     \
                        simsimd_##input_type##_t val = (src && ki < k) ? src[ki] : 0;                           \
                        dst_block[group * (16 * (interleave)) + col * (interleave) + i] = val;                  \
                    }                                                                                           \
                }                                                                                               \
            }                                                                                                   \
        }                                                                                                       \
    }

/**
 *  @brief Macro to generate serial matmul function using tiled packed B (scalar fallback).
 *
 *  Computes C = A × Bᵀ where B is pre-packed in tiled row-major format.
 *  Processes tiles in order that maximizes cache reuse: k-tiles outer for A locality.
 *  Uses MKL-style naming: simsimd_dots_{input}{input}{output}_{suffix}
 */
#define SIMSIMD_MAKE_DOTS_SERIAL_PACKED(suffix, input_type, accumulator_type, output_type, load_and_convert, tile_k) \
    SIMSIMD_PUBLIC void simsimd_dots_##input_type##input_type##output_type##_##suffix(                               \
        simsimd_##input_type##_t const *a, void const *b_packed, simsimd_##output_type##_t *c, simsimd_size_t m,     \
        simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride) {                      \
                                                                                                                     \
        simsimd_size_t const tile_n = SIMSIMD_DOTS_SERIAL_TILE_N;                                                    \
        simsimd_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                                    \
        simsimd_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                                    \
        simsimd_size_t const tile_size = tile_n * tile_k;                                                            \
                                                                                                                     \
        simsimd_##input_type##_t const *packed =                                                                     \
            (simsimd_##input_type##_t const *)((char const *)b_packed + sizeof(simsimd_dots_packed_header_t));       \
                                                                                                                     \
        /* Zero output matrix */                                                                                     \
        for (simsimd_size_t mi = 0; mi < m; ++mi) {                                                                  \
            simsimd_##output_type##_t *c_row = (simsimd_##output_type##_t *)((char *)c + mi * c_stride);             \
            for (simsimd_size_t ni = 0; ni < n; ++ni) c_row[ni] = 0;                                                 \
        }                                                                                                            \
                                                                                                                     \
        /* Process k-tiles in outer loop for better A reuse */                                                       \
        for (simsimd_size_t kt = 0; kt < k_tiles; ++kt) {                                                            \
            simsimd_size_t const k_start = kt * tile_k;                                                              \
            simsimd_size_t const k_end = (k_start + tile_k < k) ? (k_start + tile_k) : k;                            \
            simsimd_size_t const k_len = k_end - k_start;                                                            \
                                                                                                                     \
            for (simsimd_size_t mi = 0; mi < m; ++mi) {                                                              \
                simsimd_##input_type##_t const *a_row = (simsimd_##input_type##_t const *)((char const *)a +         \
                                                                                           mi * a_stride);           \
                simsimd_##output_type##_t *c_row = (simsimd_##output_type##_t *)((char *)c + mi * c_stride);         \
                                                                                                                     \
                for (simsimd_size_t nt = 0; nt < n_tiles; ++nt) {                                                    \
                    simsimd_size_t const n_start = nt * tile_n;                                                      \
                    simsimd_size_t const n_end = (n_start + tile_n < n) ? (n_start + tile_n) : n;                    \
                                                                                                                     \
                    simsimd_size_t const tile_idx = kt * n_tiles + nt;                                               \
                    simsimd_##input_type##_t const *tile = packed + tile_idx * tile_size;                            \
                                                                                                                     \
                    /* Compute partial dot products for this tile */                                                 \
                    for (simsimd_size_t j = n_start; j < n_end; ++j) {                                               \
                        simsimd_size_t const row_in_tile = j - n_start;                                              \
                        simsimd_##input_type##_t const *b_row = tile + row_in_tile * tile_k;                         \
                                                                                                                     \
                        simsimd_##accumulator_type##_t sum = 0;                                                      \
                        for (simsimd_size_t ki = 0; ki < k_len; ++ki) {                                              \
                            simsimd_##accumulator_type##_t a_val = load_and_convert(a_row + k_start + ki);           \
                            simsimd_##accumulator_type##_t b_val = load_and_convert(b_row + ki);                     \
                            sum += a_val * b_val;                                                                    \
                        }                                                                                            \
                        c_row[j] += (simsimd_##output_type##_t)sum;                                                  \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
        }                                                                                                            \
    }

// Serial packed implementations for BF16 (32 elements per 64-byte tile row)
SIMSIMD_MAKE_DOTS_SERIAL_PACKED_SIZE(serial, bf16, f32, SIMSIMD_DOTS_SERIAL_TILE_K_BF16)
SIMSIMD_MAKE_DOTS_SERIAL_PACK(serial, bf16, f32, SIMSIMD_DOTS_SERIAL_TILE_K_BF16)
SIMSIMD_MAKE_DOTS_SERIAL_PACKED(serial, bf16, f32, f32, SIMSIMD_BF16_TO_F32, SIMSIMD_DOTS_SERIAL_TILE_K_BF16)

// Serial packed implementations for I8 (64 elements per 64-byte tile row)
SIMSIMD_MAKE_DOTS_SERIAL_PACKED_SIZE(serial, i8, i32, SIMSIMD_DOTS_SERIAL_TILE_K_I8)
SIMSIMD_MAKE_DOTS_SERIAL_PACK(serial, i8, i32, SIMSIMD_DOTS_SERIAL_TILE_K_I8)
SIMSIMD_MAKE_DOTS_SERIAL_PACKED(serial, i8, i32, i32, SIMSIMD_DEREFERENCE, SIMSIMD_DOTS_SERIAL_TILE_K_I8)

/*  Serial compact functions: simple scalar implementations for post-matmul conversion.
 *  These work on any platform without SIMD requirements.
 */

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride, writes BF16 tightly packed (stride = n * sizeof(bf16)).
 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_serial( //
    void *c, simsimd_size_t m, simsimd_size_t n,      //
    simsimd_size_t c_stride) {

    simsimd_size_t const c_stride_f32 = c_stride / sizeof(simsimd_f32_t);
    simsimd_f32_t const *c_f32 = (simsimd_f32_t const *)c;
    simsimd_bf16_t *c_bf16 = (simsimd_bf16_t *)c;

    for (simsimd_size_t row = 0; row < m; row++) {
        simsimd_f32_t const *src_row = c_f32 + row * c_stride_f32;
        simsimd_bf16_t *dst_row = c_bf16 + row * n;
        for (simsimd_size_t col = 0; col < n; col++) { simsimd_f32_to_bf16(src_row + col, dst_row + col); }
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 / sqrt(a_norm[i] * b_norm[j])
 *  Output is tightly packed (stride = n * sizeof(i8)).
 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i8_serial(  //
    void *c, simsimd_size_t m, simsimd_size_t n, //
    simsimd_size_t c_stride,                     //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms) {

    simsimd_size_t const c_stride_i32 = c_stride / sizeof(simsimd_i32_t);
    simsimd_i32_t const *c_i32 = (simsimd_i32_t const *)c;
    simsimd_i8_t *c_i8 = (simsimd_i8_t *)c;

    for (simsimd_size_t row = 0; row < m; row++) {
        simsimd_i32_t const *src_row = c_i32 + row * c_stride_i32;
        simsimd_i8_t *dst_row = c_i8 + row * n;

        simsimd_f32_t a_norm_f32 = (simsimd_f32_t)a_squared_norms[row];
        simsimd_f32_t a_rsqrt = (a_norm_f32 > 0) ? (1.0f / SIMSIMD_SQRT(a_norm_f32)) : 0.0f;

        for (simsimd_size_t col = 0; col < n; col++) {
            simsimd_f32_t b_norm_f32 = (simsimd_f32_t)b_squared_norms[col];
            simsimd_f32_t b_rsqrt = (b_norm_f32 > 0) ? (1.0f / SIMSIMD_SQRT(b_norm_f32)) : 0.0f;

            simsimd_f32_t normalized = (simsimd_f32_t)src_row[col] * 127.0f * a_rsqrt * b_rsqrt;
            simsimd_i32_t clamped = (simsimd_i32_t)normalized;
            if (clamped < -128) clamped = -128;
            if (clamped > 127) clamped = 127;
            dst_row[col] = (simsimd_i8_t)clamped;
        }
    }
}

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_SVE

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE
#endif // SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

/** @brief Load 16 F32 values into a 512-bit vector. */
SIMSIMD_INTERNAL void _simsimd_load_b512_f32(simsimd_f32_t const *src, simsimd_b512_vec_t *dst) {
    dst->zmm_ps = _mm512_loadu_ps(src);
}

// F32 GEMM packing functions (interleave=1, k_align=1)
SIMSIMD_MAKE_DOTS_PACKED_SIZE(skylake, f32, f32, 1)
SIMSIMD_MAKE_DOTS_PACK(skylake, f32, f32, 1, 1)

// F32 GEMM: MR=8, KC=256, MC=128, NC=2048
// Uses vfmadd231ps with {1to16} embedded broadcast for efficient scalar×vector FMA
SIMSIMD_MAKE_DOTS_MRx16(f32f32f32_skylake, f32, f32, simsimd_dot_outer_f32x2_state_1x16_skylake_t,
                        simsimd_dot_outer_f32x2_init_1x16_skylake, simsimd_dot_outer_f32x2_update_1x16_skylake,
                        simsimd_dot_outer_f32x2_finalize_1x16_skylake,
                        /*k_tile=*/2, /*MR=*/8, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), apply_to = function)

    /** @brief Load 32 BF16 values into a 512-bit vector. */
    SIMSIMD_INTERNAL void _simsimd_load_b512_bf16(simsimd_bf16_t const *src, simsimd_b512_vec_t *dst) {
    dst->zmm = _mm512_loadu_epi16(src);
}

// BF16 GEMM packing functions (interleave=2 for VDPBF16PS, k_align=16)
SIMSIMD_MAKE_DOTS_PACKED_SIZE(genoa, bf16, f32, 16)
SIMSIMD_MAKE_DOTS_PACK(genoa, bf16, f32, 2, 16)

// Compact function: F32→BF16 conversion (reuses serial implementation logic)
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_genoa(void *c, simsimd_size_t m, simsimd_size_t n,
                                                    simsimd_size_t c_stride) {
    simsimd_dots_bf16bf16bf16_serial(c, m, n, c_stride);
}

// BF16 GEMM: MR=8, k_tile=4, nr_acc=4 for 4× throughput
SIMSIMD_MAKE_DOTS(bf16bf16f32_genoa, bf16, f32, simsimd_dot_outer_bf16x4_state_1x16_genoa_t,
                  simsimd_dot_outer_bf16x4_init_1x16_genoa, simsimd_dot_outer_bf16x4_update_1x16_genoa,
                  simsimd_dot_outer_bf16x4_finalize_1x16_genoa,
                  /*k_tile=*/4, /*MR=*/8, /*nr_acc=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE_AMX
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16", "amx-tile", "amx-bf16", "amx-int8")
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512fp16,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)

/*  AMX tile dimensions (Intel Sapphire Rapids):
 *
 *    BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *    INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  Output pattern: 2×2 tile layout produces 32×32 output blocks.
 *  Register allocation:
 *    TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *    TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *    TMM4-7: C accumulator tiles (2×2 output grid)
 *
 *  Performance characteristics (single-threaded):
 *    - BF16 peak: ~500 GFLOPS per core (2× FP32 throughput)
 *    - INT8 peak: ~1000 GOPS per core (4× FP32 throughput)
 *    - Memory bandwidth: ~80 GB/s DDR5 per core
 *    - Optimal K dimension: multiples of 32 (BF16) or 64 (INT8)
 *
 *  Acceleration opportunities:
 *    - Pre-pack B matrix once for repeated inference (avoids runtime reordering)
 *    - Morton Z-curve tile ordering improves L2 cache hit rate by 5-25%
 *    - Partition A rows across threads for parallel execution
 *    - Use streaming stores for large C matrices to avoid cache pollution
 */

/*  Morton Z-curve encoding for cache-friendly tile traversal.
 *  Uses BMI2 PDEP instruction for fast (2-3 cycle) bit interleaving.
 *  Interleaves bits of (tile_row, tile_col) to produce Z-curve index.
 */
SIMSIMD_INTERNAL simsimd_u64_t _simsimd_morton_encode_sapphire_amx(simsimd_u32_t tile_row, simsimd_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/*  Configure AMX tile registers.
 *  Called once per kernel invocation (idempotent within a thread).
 *  Sets all 8 tiles to standard 16 rows × 64 bytes layout.
 *
 *  Note: OS permission for AMX must be requested before using AMX instructions.
 *  Call `simsimd_flush_denormals(simsimd_capabilities())` once per thread
 *  before using any Sapphire matmul functions.
 */
SIMSIMD_INTERNAL void _simsimd_amx_tile_configure_sapphire_amx(void) {
    SIMSIMD_ALIGN64 simsimd_u8_t tile_config[64] = {0};
    tile_config[0] = 1; // palette 1 (standard tile configuration)

    simsimd_u16_t *bytes_per_row = (simsimd_u16_t *)&tile_config[16];
    simsimd_u8_t *rows_per_tile = &tile_config[48];

    for (int tile_id = 0; tile_id < 8; tile_id++) {
        rows_per_tile[tile_id] = 16; // 16 rows per tile
        bytes_per_row[tile_id] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/*  Compiler memory barrier to ensure stores complete before AMX tile loads.
 *  AMX _tile_loadd reads from memory written by AVX-512 stores. Without this barrier,
 *  the compiler may reorder or optimize away the stores, causing _tile_loadd to read stale data.
 *  This is a compiler-only fence (no CPU fence needed - same core, same memory).
 */
SIMSIMD_INTERNAL void _simsimd_compiler_barrier_sapphire_amx(void) { __asm__ volatile("" ::: "memory"); }

/*  AVX-512 masked load for A tile (BF16): loads up to 16 rows × 32 cols into aligned buffer.
 *  Uses masked loads to handle edge tiles without element-wise loops.
 *  Includes memory barrier to ensure stores complete before subsequent _tile_loadd.
 */
SIMSIMD_INTERNAL void _simsimd_load_a_tile_bf16_masked(            //
    simsimd_bf16_t const *src, simsimd_size_t src_stride_elements, //
    simsimd_size_t valid_rows, simsimd_size_t valid_cols, simsimd_bf16_t *dst /*[16][32]*/) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (simsimd_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi16(col_mask, src + r * src_stride_elements);
            _mm512_store_si512((__m512i *)(dst + r * 32), row);
        }
        else { _mm512_store_si512((__m512i *)(dst + r * 32), zero); }
    }
    _simsimd_compiler_barrier_sapphire_amx();
}

/*  AVX-512 masked load for A tile (I8): loads up to 16 rows × 64 cols into aligned buffer.
 *  Includes memory barrier to ensure stores complete before subsequent _tile_loadd.
 */
SIMSIMD_INTERNAL void _simsimd_load_a_tile_i8_masked(   //
    simsimd_i8_t const *src, simsimd_size_t src_stride, //
    simsimd_size_t valid_rows, simsimd_size_t valid_cols, simsimd_i8_t *dst /*[16][64]*/) {

    __mmask64 col_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (simsimd_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi8(col_mask, src + r * src_stride);
            _mm512_store_si512((__m512i *)(dst + r * 64), row);
        }
        else { _mm512_store_si512((__m512i *)(dst + r * 64), zero); }
    }
    _simsimd_compiler_barrier_sapphire_amx();
}

/*  AVX-512 masked store for C tile (F32): stores up to 16 rows × 16 cols from aligned buffer.
 */
SIMSIMD_INTERNAL void _simsimd_store_c_tile_f32_masked(                                   //
    simsimd_f32_t const *src /*[16][16]*/, simsimd_f32_t *dst, simsimd_size_t dst_stride, //
    simsimd_size_t valid_rows, simsimd_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (simsimd_size_t r = 0; r < valid_rows; r++) {
        __m512 row = _mm512_load_ps(src + r * 16);
        _mm512_mask_storeu_ps(dst + r * dst_stride, col_mask, row);
    }
}

/*  AVX-512 masked store for C tile (I32): stores up to 16 rows × 16 cols from aligned buffer.
 */
SIMSIMD_INTERNAL void _simsimd_store_c_tile_i32_masked(                                   //
    simsimd_i32_t const *src /*[16][16]*/, simsimd_i32_t *dst, simsimd_size_t dst_stride, //
    simsimd_size_t valid_rows, simsimd_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (simsimd_size_t r = 0; r < valid_rows; r++) {
        __m512i row = _mm512_load_si512((__m512i const *)(src + r * 16));
        _mm512_mask_storeu_epi32(dst + r * dst_stride, col_mask, row);
    }
}

/*  AVX-512 edge matmul for BF16 → F32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 *
 *  B is stored in row-major as b_edge[row * k + col] where row is the N index.
 *  This computes: C[i,j] = sum_over_k(A[i,k] * B[j,k])
 */
SIMSIMD_INTERNAL void _simsimd_dots_bf16bf16f32_avx512_edge(                 //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b_edge, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,                    //
    simsimd_size_t a_stride_elements, simsimd_size_t b_stride_k, simsimd_size_t c_stride_elements) {

    // Process each output row
    for (simsimd_size_t i = 0; i < m; i++) {
        simsimd_bf16_t const *a_row = a + i * a_stride_elements;

        // Process output columns in chunks of 16 (AVX-512 width for F32)
        for (simsimd_size_t j = 0; j < n; j += 16) {
            simsimd_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512 acc = _mm512_setzero_ps();

            // Dot product over K dimension - process 2 at a time for BF16 pairs
            for (simsimd_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to F32
                simsimd_f32_t a_val = (simsimd_f32_t)a_row[kk];
                __m512 a_bc = _mm512_set1_ps(a_val);

                // Gather B[j:j+16, k] - each B row is stored with stride b_stride_k
                // b_edge[row * b_stride_k + col] where row in [0,n), col in [0,k)
                SIMSIMD_ALIGN64 simsimd_f32_t b_vals[16];
                for (simsimd_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (simsimd_f32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (simsimd_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0.0f;

                __m512 b_vec = _mm512_load_ps(b_vals);
                acc = _mm512_fmadd_ps(a_bc, b_vec, acc);
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_ps(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  AVX-512 edge matmul for I8 → I32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 */
SIMSIMD_INTERNAL void _simsimd_dots_i8i8i32_avx512_edge(                 //
    simsimd_i8_t const *a, simsimd_i8_t const *b_edge, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,                //
    simsimd_size_t a_stride, simsimd_size_t b_stride_k, simsimd_size_t c_stride_elements) {

    // Process each output row
    for (simsimd_size_t i = 0; i < m; i++) {
        simsimd_i8_t const *a_row = a + i * a_stride;

        // Process output columns in chunks of 16 (AVX-512 width for I32)
        for (simsimd_size_t j = 0; j < n; j += 16) {
            simsimd_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512i acc = _mm512_setzero_si512();

            // Dot product over K dimension
            for (simsimd_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to I32
                simsimd_i32_t a_val = (simsimd_i32_t)a_row[kk];
                __m512i a_bc = _mm512_set1_epi32(a_val);

                // Gather B[j:j+16, k]
                SIMSIMD_ALIGN64 simsimd_i32_t b_vals[16];
                for (simsimd_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (simsimd_i32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (simsimd_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0;

                __m512i b_vec = _mm512_load_si512((__m512i const *)b_vals);
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(a_bc, b_vec));
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_epi32(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  BF16 packed buffer size: header + all tiles for full N rows + N edge.
 *  Hybrid layout:
 *    - Tiles include K remainder (zero-padded) for AMX to handle full dot products
 *    - N edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 32;
    simsimd_size_t const tile_bytes = 512 * sizeof(simsimd_bf16_t); // 16×32×2 = 1KB

    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    simsimd_size_t size = sizeof(simsimd_dots_packed_header_t);

    // All tiles for full N rows (Morton-ordered, pair-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(simsimd_bf16_t);

    return size;
}

/*  I8 packed buffer size: header + all tiles for full N rows + N edge.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_i8i8i32_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 64;
    simsimd_size_t const tile_bytes = 1024 * sizeof(simsimd_i8_t); // 16×64×1 = 1KB

    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    simsimd_size_t size = sizeof(simsimd_dots_packed_header_t);

    // All tiles for full N rows (Morton-ordered, quad-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(simsimd_i8_t);

    return size;
}

/*  Pack BF16 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: Morton Z-curve ordered, pair-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX BF16 tile format: for TDPBF16PS, B tile should have elements arranged so that
 *  consecutive pairs of columns are interleaved by rows:
 *    [col0_row0, col1_row0, col0_row1, col1_row1, ..., col0_row15, col1_row15,
 *     col2_row0, col3_row0, col2_row1, col3_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_pack_sapphire_amx(  //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 32;
    simsimd_size_t const tile_elements = 512;
    simsimd_size_t const tile_bytes = tile_elements * sizeof(simsimd_bf16_t);
    simsimd_size_t const b_stride_elements = b_stride / sizeof(simsimd_bf16_t);

    // Compute layout dimensions
    simsimd_size_t const num_n_tiles = n / tile_rows;
    simsimd_size_t const num_k_tiles = (k + tile_cols - 1) / tile_cols;
    simsimd_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    simsimd_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    simsimd_dots_packed_header_t *header = (simsimd_dots_packed_header_t *)b_packed;
    header->full_n_tiles = (simsimd_u32_t)num_n_tiles;
    header->full_k_tiles = (simsimd_u32_t)num_k_tiles;
    header->n_edge_rows = (simsimd_u32_t)n_remainder_rows;

    // Compute memory region offsets
    simsimd_size_t const tiles_offset = sizeof(simsimd_dots_packed_header_t);
    simsimd_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (simsimd_u32_t)n_edge_offset;

    // Pointers to packed data regions
    simsimd_bf16_t *tiles_ptr = (simsimd_bf16_t *)((char *)b_packed + tiles_offset);
    simsimd_bf16_t *n_edge_ptr = (simsimd_bf16_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (simsimd_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension,
    // which is critical for cache efficiency in the compute kernel.
    for (simsimd_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            simsimd_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            simsimd_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            simsimd_size_t const src_row_start = n_tile * tile_rows;
            simsimd_size_t const src_col_start = k_tile * tile_cols;
            simsimd_size_t const cols_to_pack = (src_col_start + tile_cols <= k) ? tile_cols : (k - src_col_start);

            // Pack with pair-interleaving as required by TDPBF16PS instruction.
            // AMX expects: [col0_row0, col1_row0, col0_row1, col1_row1, col2_row0, col3_row0, ...]
            // Formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
            for (simsimd_size_t row = 0; row < tile_rows; row++) {
                for (simsimd_size_t col = 0; col < cols_to_pack; col++) {
                    simsimd_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    simsimd_size_t const dst_idx = (col / 2) * 32 + row * 2 + (col % 2);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        simsimd_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (simsimd_size_t row = 0; row < n_remainder_rows; row++) {
            for (simsimd_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  Pack I8 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: linearly ordered, quad-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_pack_sapphire_amx(    //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 64;
    simsimd_size_t const tile_elements = 1024;
    simsimd_size_t const tile_bytes = tile_elements * sizeof(simsimd_i8_t);

    // Compute layout dimensions
    simsimd_size_t const num_n_tiles = n / tile_rows;
    simsimd_size_t const num_k_tiles = (k + tile_cols - 1) / tile_cols;
    simsimd_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    simsimd_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    simsimd_dots_packed_header_t *header = (simsimd_dots_packed_header_t *)b_packed;
    header->full_n_tiles = (simsimd_u32_t)num_n_tiles;
    header->full_k_tiles = (simsimd_u32_t)num_k_tiles;
    header->n_edge_rows = (simsimd_u32_t)n_remainder_rows;

    // Compute memory region offsets
    simsimd_size_t const tiles_offset = sizeof(simsimd_dots_packed_header_t);
    simsimd_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (simsimd_u32_t)n_edge_offset;

    // Pointers to packed data regions
    simsimd_i8_t *tiles_ptr = (simsimd_i8_t *)((char *)b_packed + tiles_offset);
    simsimd_i8_t *n_edge_ptr = (simsimd_i8_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (simsimd_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension.
    for (simsimd_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            simsimd_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            simsimd_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            simsimd_size_t const src_row_start = n_tile * tile_rows;
            simsimd_size_t const src_col_start = k_tile * tile_cols;
            simsimd_size_t const cols_to_pack = (src_col_start + tile_cols <= k) ? tile_cols : (k - src_col_start);

            // Pack with quad-interleaving as required by TDPBSSD instruction.
            // AMX expects: [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, ...]
            // Formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
            for (simsimd_size_t row = 0; row < tile_rows; row++) {
                for (simsimd_size_t col = 0; col < cols_to_pack; col++) {
                    simsimd_size_t const src_idx = (src_row_start + row) * b_stride + src_col_start + col;
                    simsimd_size_t const dst_idx = (col / 4) * 64 + row * 4 + (col % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        simsimd_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (simsimd_size_t row = 0; row < n_remainder_rows; row++) {
            for (simsimd_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride + col];
            }
        }
    }
}

/*  BF16 → F32 matmul (aligned path): Direct tile loads/stores when stride >= 64 bytes.
 *
 *  Optimized with:
 *  - Nc=2 panel blocking: process 2 N-blocks (64 columns) at a time to maximize B-tile reuse
 *  - Software pipelining: overlap tile loads with compute operations
 *  - Linear B indexing: sequential memory access along K dimension
 *
 *  AMX tile usage:
 *    TMM0-1: A tiles (2 rows of 16×32 from current M-block)
 *    TMM2-3: B tiles (2 tiles from current N-block)
 *    TMM4-7: C accumulators (2×2 = 4 output tiles of 16×16 each)
 */
SIMSIMD_INTERNAL void _simsimd_dots_bf16bf16f32_sapphire_aligned(    //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Read packed B header
    simsimd_dots_packed_header_t const *header = (simsimd_dots_packed_header_t const *)b_packed;
    simsimd_size_t const num_n_tiles = header->full_n_tiles;
    simsimd_size_t const num_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_remainder_rows = header->n_edge_rows;

    // Pointers to packed data regions
    simsimd_bf16_t const *b_tiles = (simsimd_bf16_t const *)((char const *)b_packed +
                                                             sizeof(simsimd_dots_packed_header_t));
    simsimd_bf16_t const *n_edge_ptr = (simsimd_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    // Constants for BF16 AMX tiles
    simsimd_size_t const tile_k_cols = 32;    // K-dimension of one tile
    simsimd_size_t const tile_elements = 512; // 16 rows × 32 cols

    // Stride conversions
    simsimd_size_t const a_stride_bf16 = a_stride / sizeof(simsimd_bf16_t);
    simsimd_size_t const c_stride_f32 = c_stride / sizeof(simsimd_f32_t);

    // Block dimensions
    simsimd_size_t const num_m_blocks = m / 32;          // Each M-block = 32 rows (2 tiles)
    simsimd_size_t const num_n_blocks = num_n_tiles / 2; // Each N-block = 32 cols (2 tiles)
    simsimd_size_t const full_n_cols = num_n_tiles * 16;

    // Nc=2 panel size: process 2 N-blocks (64 columns) per outer iteration
    // This keeps B tiles hot in L2 while streaming through A rows
    simsimd_size_t const panel_size = 2;

    // AMX: Full 32×32 output blocks with Nc=2 blocking and software pipelining
    if (num_m_blocks > 0 && num_n_blocks > 0 && num_k_tiles > 0) {
        _simsimd_amx_tile_configure_sapphire_amx();

        // Outer loop: N-panels of size Nc=2
        for (simsimd_size_t n_panel_start = 0; n_panel_start < num_n_blocks; n_panel_start += panel_size) {
            simsimd_size_t const n_panel_end = (n_panel_start + panel_size < num_n_blocks)
                                                   ? (n_panel_start + panel_size)
                                                   : num_n_blocks;

            // Middle loop: all M-blocks (B tiles stay hot for each M-block)
            for (simsimd_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
                simsimd_size_t const m_row = m_block * 32;

                // A tile base addresses for this M-block
                simsimd_bf16_t const *a_row0 = a + m_row * a_stride_bf16;
                simsimd_bf16_t const *a_row1 = a + (m_row + 16) * a_stride_bf16;

                // Inner loop: N-blocks within current panel
                for (simsimd_size_t n_block = n_panel_start; n_block < n_panel_end; n_block++) {
                    simsimd_size_t const n_col = n_block * 32;

                    // B tile base indices for this N-block (linear layout)
                    simsimd_size_t const b_n0_base = (n_block * 2) * num_k_tiles;     // First N-tile
                    simsimd_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles; // Second N-tile

                    // Zero accumulators
                    _tile_zero(4);
                    _tile_zero(5);
                    _tile_zero(6);
                    _tile_zero(7);

                    // Software-pipelined K-loop
                    if (num_k_tiles > 1) {
                        // Prologue: load first tiles
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_elements, 64);

                        // Main loop: compute current, load next
                        for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles - 1; k_tile++) {
                            simsimd_size_t const k_offset = k_tile * tile_k_cols;
                            simsimd_size_t const next_k_offset = (k_tile + 1) * tile_k_cols;

                            // Compute A0×B0, load A1 and B1
                            _tile_dpbf16ps(4, 0, 2);
                            _tile_loadd(1, a_row1 + k_offset, (int)a_stride);
                            _tile_loadd(3, b_tiles + (b_n1_base + k_tile) * tile_elements, 64);

                            // Compute A1×B0, A0×B1, A1×B1
                            _tile_dpbf16ps(6, 1, 2);
                            _tile_dpbf16ps(5, 0, 3);
                            _tile_dpbf16ps(7, 1, 3);

                            // Load next iteration's A0 and B0
                            _tile_loadd(0, a_row0 + next_k_offset, (int)a_stride);
                            _tile_loadd(2, b_tiles + (b_n0_base + k_tile + 1) * tile_elements, 64);
                        }

                        // Epilogue: last K iteration
                        simsimd_size_t const last_k = num_k_tiles - 1;
                        simsimd_size_t const last_k_offset = last_k * tile_k_cols;

                        _tile_dpbf16ps(4, 0, 2);
                        _tile_loadd(1, a_row1 + last_k_offset, (int)a_stride);
                        _tile_dpbf16ps(6, 1, 2);
                        _tile_loadd(3, b_tiles + (b_n1_base + last_k) * tile_elements, 64);
                        _tile_dpbf16ps(5, 0, 3);
                        _tile_dpbf16ps(7, 1, 3);
                    }
                    else {
                        // Single K-tile: no pipelining needed
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(1, a_row1, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_elements, 64);
                        _tile_loadd(3, b_tiles + b_n1_base * tile_elements, 64);
                        _tile_dpbf16ps(4, 0, 2);
                        _tile_dpbf16ps(5, 0, 3);
                        _tile_dpbf16ps(6, 1, 2);
                        _tile_dpbf16ps(7, 1, 3);
                    }

                    // Store 2×2 output block directly to C
                    simsimd_f32_t *c_block = c + m_row * c_stride_f32 + n_col;
                    _tile_stored(4, c_block, (int)c_stride);
                    _tile_stored(5, c_block + 16, (int)c_stride);
                    _tile_stored(6, c_block + 16 * c_stride_f32, (int)c_stride);
                    _tile_stored(7, c_block + 16 * c_stride_f32 + 16, (int)c_stride);
                }
            }
        }

        _tile_release();
    }

    // AVX-512: N-remainder rows (rows beyond full N-tiles)
    if (n_remainder_rows > 0) {
        _simsimd_dots_bf16bf16f32_avx512_edge(a, n_edge_ptr, c + full_n_cols, m, n_remainder_rows, k, a_stride_bf16, k,
                                              c_stride_f32);
    }

    // AMX: M-remainder rows (rows beyond full M-blocks) for full N-tiles
    if (m > num_m_blocks * 32 && num_n_tiles > 0) {
        simsimd_size_t const m_remainder_start = num_m_blocks * 32;
        simsimd_size_t const m_remainder_count = m - m_remainder_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        // Process each N-tile individually for M-remainder
        for (simsimd_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
            simsimd_size_t const n_col = n_tile * 16;

            _tile_zero(4);
            _tile_zero(6);

            // Staging buffers for partial A tiles
            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

            simsimd_size_t const rows_upper = (m_remainder_count > 16) ? 16 : m_remainder_count;
            simsimd_size_t const rows_lower = (m_remainder_count > 16) ? m_remainder_count - 16 : 0;

            for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                simsimd_size_t const k_offset = k_tile * tile_k_cols;
                simsimd_size_t const k_valid = (k_offset + tile_k_cols <= k) ? tile_k_cols : (k - k_offset);

                // Load partial A tiles with masking
                _simsimd_load_a_tile_bf16_masked(a + m_remainder_start * a_stride_bf16 + k_offset, a_stride_bf16,
                                                 rows_upper, k_valid, (simsimd_bf16_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_bf16_masked(a + (m_remainder_start + 16) * a_stride_bf16 + k_offset,
                                                     a_stride_bf16, rows_lower, k_valid,
                                                     (simsimd_bf16_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // Linear B tile index
                simsimd_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                _tile_loadd(2, b_tiles + b_tile_idx * tile_elements, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            // Store with masking for partial rows
            SIMSIMD_ALIGN64 simsimd_f32_t c_tile_buf[16][16];

            _tile_stored(4, c_tile_buf, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile_buf, c + m_remainder_start * c_stride_f32 + n_col,
                                             c_stride_f32, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile_buf, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile_buf,
                                                 c + (m_remainder_start + 16) * c_stride_f32 + n_col, c_stride_f32,
                                                 rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: M-remainder × N-remainder corner
    if (m > num_m_blocks * 32 && n_remainder_rows > 0) {
        simsimd_size_t const m_remainder_start = num_m_blocks * 32;
        simsimd_size_t const m_remainder_count = m - m_remainder_start;

        _simsimd_dots_bf16bf16f32_avx512_edge(a + m_remainder_start * a_stride_bf16, n_edge_ptr,
                                              c + m_remainder_start * c_stride_f32 + full_n_cols, m_remainder_count,
                                              n_remainder_rows, k, a_stride_bf16, k, c_stride_f32);
    }
}

/*  BF16 → F32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
SIMSIMD_INTERNAL void _simsimd_dots_bf16bf16f32_sapphire_misaligned( //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Read header for hybrid layout
    simsimd_dots_packed_header_t const *header = (simsimd_dots_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;

    simsimd_bf16_t const *tiles_ptr = (simsimd_bf16_t const *)((char const *)b_packed +
                                                               sizeof(simsimd_dots_packed_header_t));
    simsimd_bf16_t const *n_edge_ptr = (simsimd_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    simsimd_size_t const tile_cols_bf16 = 32;
    simsimd_size_t const tile_elements_bf16 = 512;

    simsimd_size_t const a_stride_elements = a_stride / sizeof(simsimd_bf16_t);
    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_f32_t);
    simsimd_size_t const full_n = full_n_tiles * 16;
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = full_n_tiles / 2;
    simsimd_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // Stack buffers for tile I/O
    SIMSIMD_ALIGN64 simsimd_bf16_t a_buf_upper[16][32];
    SIMSIMD_ALIGN64 simsimd_bf16_t a_buf_lower[16][32];
    SIMSIMD_ALIGN64 simsimd_f32_t c_buf[16][16];

    // AMX: Full 32×32 blocks through buffers
    if (full_m_blocks > 0 && full_n_blocks > 0 && full_k_tiles > 0) {
        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
            simsimd_size_t const row_block = bi * 32;

            for (simsimd_size_t bj = 0; bj < full_n_blocks; bj++) {
                simsimd_size_t const col_block = bj * 32;

                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);

                simsimd_size_t const b_tile_n0 = bj * 2;
                simsimd_size_t const b_tile_n1 = bj * 2 + 1;

                for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                    simsimd_size_t const k_offset = bk * tile_cols_bf16;

                    // Load A through buffers using AVX-512
                    for (simsimd_size_t r = 0; r < 16; r++) {
                        simsimd_bf16_t const *src_upper = a + (row_block + r) * a_stride_elements + k_offset;
                        simsimd_bf16_t const *src_lower = a + (row_block + 16 + r) * a_stride_elements + k_offset;
                        __m512i upper_row = _mm512_loadu_si512((__m512i const *)src_upper);
                        __m512i lower_row = _mm512_loadu_si512((__m512i const *)src_lower);
                        _mm512_store_si512((__m512i *)a_buf_upper[r], upper_row);
                        _mm512_store_si512((__m512i *)a_buf_lower[r], lower_row);
                    }
                    __asm__ volatile("" ::: "memory");

                    _tile_loadd(0, a_buf_upper, 64);
                    _tile_loadd(1, a_buf_lower, 64);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0,
                                                                                     (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                    simsimd_size_t morton_idx1 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1,
                                                                                     (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    simsimd_bf16_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_bf16;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }

                // Store C through buffers using AVX-512
                _tile_stored(4, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + r) * c_stride_elements + col_block, row);
                }

                _tile_stored(5, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + r) * c_stride_elements + col_block + 16, row);
                }

                _tile_stored(6, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + 16 + r) * c_stride_elements + col_block, row);
                }

                _tile_stored(7, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + 16 + r) * c_stride_elements + col_block + 16, row);
                }
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0) {
        _simsimd_dots_bf16bf16f32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride_elements, k,
                                              c_stride_elements);
    }

    // AMX: M edge rows for full N tiles (through buffers)
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t tj = 0; tj < full_n_tiles; tj++) {
            simsimd_size_t const col_block = tj * 16;
            simsimd_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            // Zero buffers for edge
            for (simsimd_size_t r = 0; r < 16; r++) {
                _mm512_store_si512((__m512i *)a_buf_upper[r], _mm512_setzero_si512());
                _mm512_store_si512((__m512i *)a_buf_lower[r], _mm512_setzero_si512());
            }

            simsimd_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            simsimd_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                simsimd_size_t const k_offset = bk * tile_cols_bf16;
                simsimd_size_t const k_valid = (k_offset + tile_cols_bf16 <= k) ? tile_cols_bf16 : (k - k_offset);

                _simsimd_load_a_tile_bf16_masked(a + m_edge_start * a_stride_elements + k_offset, a_stride_elements,
                                                 rows_upper, k_valid, (simsimd_bf16_t *)a_buf_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_bf16_masked(a + (m_edge_start + 16) * a_stride_elements + k_offset,
                                                     a_stride_elements, rows_lower, k_valid,
                                                     (simsimd_bf16_t *)a_buf_lower);
                }

                _tile_loadd(0, a_buf_upper, 64);
                _tile_loadd(1, a_buf_lower, 64);

                simsimd_size_t morton_idx0 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0,
                                                                                 (simsimd_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                simsimd_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_buf, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_buf, c + m_edge_start * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_buf, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_buf,
                                                 c + (m_edge_start + 16) * c_stride_elements + col_block,
                                                 c_stride_elements, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_count = m - m_edge_start;

        _simsimd_dots_bf16bf16f32_avx512_edge(a + m_edge_start * a_stride_elements, n_edge_ptr,
                                              c + m_edge_start * c_stride_elements + full_n, m_edge_count, n_edge_rows,
                                              k, a_stride_elements, k, c_stride_elements);
    }
}

/*  BF16 → F32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_sapphire_amx(           //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes = 32 BF16 or 16 F32)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) _simsimd_dots_bf16bf16f32_sapphire_aligned(a, b_packed, c, m, n, k, a_stride, c_stride);
    else
        _simsimd_dots_bf16bf16f32_sapphire_misaligned(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = n * sizeof(bf16).
 */
SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,            //
    simsimd_size_t c_stride) {

    simsimd_size_t const c_stride_f32 = c_stride / sizeof(simsimd_f32_t);
    simsimd_f32_t const *c_f32 = (simsimd_f32_t const *)c;
    simsimd_bf16_t *c_bf16 = (simsimd_bf16_t *)c;

    for (simsimd_size_t row = 0; row < m; row++) {
        simsimd_f32_t const *src_row = c_f32 + row * c_stride_f32;
        simsimd_bf16_t *dst_row = c_bf16 + row * n;
        simsimd_size_t col = 0;

        // Process 16 floats at a time using AVX512-BF16
        for (; col + 16 <= n; col += 16) {
            __m512 f32_vec = _mm512_loadu_ps(src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_storeu_si256((__m256i *)(dst_row + col), (__m256i)bf16_vec);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512 f32_vec = _mm512_maskz_loadu_ps(tail_mask, src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_mask_storeu_epi16(dst_row + col, tail_mask, (__m256i)bf16_vec);
        }
    }
}

/*  I8 → I32 matmul (aligned path): Direct tile loads/stores when stride >= 64 bytes.
 *  This is the fast path - no intermediate buffers for A or C.
 *
 *  Optimization strategy:
 *  - Nc=2 panel blocking: Process 2 N-blocks (64 columns) per outer iteration
 *    to maximize B tile reuse across M-blocks
 *  - Linear B indexing: tile_index = n_tile * num_k_tiles + k_tile for sequential
 *    memory access when streaming along K dimension
 *  - Software pipelining: Overlap tile loads with compute operations
 *  - 2×2 output blocking: 4 accumulator tiles (TMM4-7) for each 32×32 C block
 */
SIMSIMD_INTERNAL void _simsimd_dots_i8i8i32_sapphire_aligned(      //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Parse packed B header
    simsimd_dots_packed_header_t const *header = (simsimd_dots_packed_header_t const *)b_packed;
    simsimd_size_t const num_n_tiles = header->full_n_tiles;    // Number of 16-column N tiles
    simsimd_size_t const num_k_tiles = header->full_k_tiles;    // Number of 64-element K tiles
    simsimd_size_t const n_edge_cols = header->n_edge_rows;     // Columns in N edge (0-15)
    simsimd_size_t const n_edge_offset = header->n_edge_offset; // Byte offset to N edge data

    // AMX I8 tile dimensions: 16 rows × 64 columns = 1024 I8 elements = 1KB
    simsimd_size_t const tile_k_elements = 64;  // K elements per tile
    simsimd_size_t const tile_byte_size = 1024; // Total bytes per packed tile

    // Pointer to packed B tiles (after 64-byte header)
    simsimd_i8_t const *b_tiles = (simsimd_i8_t const *)((char const *)b_packed + 64);
    simsimd_i8_t const *n_edge_tiles = (n_edge_offset > 0)
                                           ? (simsimd_i8_t const *)((char const *)b_packed + n_edge_offset)
                                           : NULL;

    // Dimension calculations
    simsimd_size_t const c_stride_i32 = c_stride / sizeof(simsimd_i32_t); // C stride in elements
    simsimd_size_t const num_m_blocks = m / 32;                           // Number of 32-row M blocks
    simsimd_size_t const num_n_blocks = n / 32;          // Number of 32-column N blocks (each = 2 N tiles)
    simsimd_size_t const full_n_cols = num_n_tiles * 16; // Total columns covered by full N tiles

    // Nc=2 panel size: process 2 N-blocks (64 columns) per outer iteration
    // This keeps 2 × 2 × num_k_tiles B tiles hot in L2 cache
    simsimd_size_t const panel_size = 2;

    // AMX: Full 32×32 blocks with direct I/O
    if (num_m_blocks > 0 && num_n_blocks > 0 && num_k_tiles > 0) {
        _simsimd_amx_tile_configure_sapphire_amx();

        // Outer loop: N-panels of size Nc=2
        for (simsimd_size_t n_panel_start = 0; n_panel_start < num_n_blocks; n_panel_start += panel_size) {
            simsimd_size_t const n_panel_end = (n_panel_start + panel_size < num_n_blocks)
                                                   ? (n_panel_start + panel_size)
                                                   : num_n_blocks;

            // Middle loop: all M-blocks (B tiles stay hot for each M-block)
            for (simsimd_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
                simsimd_size_t const m_row = m_block * 32; // Starting row in A and C

                // Pointer to A rows for this M-block
                simsimd_i8_t const *a_row0 = a + m_row * a_stride;        // Upper 16 rows
                simsimd_i8_t const *a_row1 = a + (m_row + 16) * a_stride; // Lower 16 rows

                // Inner loop: N-blocks within current panel
                for (simsimd_size_t n_block = n_panel_start; n_block < n_panel_end; n_block++) {
                    simsimd_size_t const n_col = n_block * 32; // Starting column in C

                    // Initialize accumulator tiles
                    _tile_zero(4); // C[row0:row0+16, col:col+16]
                    _tile_zero(5); // C[row0:row0+16, col+16:col+32]
                    _tile_zero(6); // C[row0+16:row0+32, col:col+16]
                    _tile_zero(7); // C[row0+16:row0+32, col+16:col+32]

                    // B tile base indices in packed buffer (linear layout)
                    // Linear: tile_index = n_tile * num_k_tiles + k_tile
                    simsimd_size_t const b_n0_base = (n_block * 2) * num_k_tiles;     // Left B column
                    simsimd_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles; // Right B column

                    // Software-pipelined K-loop
                    if (num_k_tiles > 1) {
                        // Prologue: Load first A and B tiles
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_byte_size, 64);

                        // Main loop: compute current tiles while loading next
                        for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles - 1; k_tile++) {
                            simsimd_size_t const k_offset = k_tile * tile_k_elements;
                            simsimd_size_t const next_k_offset = (k_tile + 1) * tile_k_elements;

                            // Compute: A0 × B0 → C[0,0]
                            _tile_dpbssd(4, 0, 2);

                            // Load: A1 (lower rows), B1 (right column)
                            _tile_loadd(1, a_row1 + k_offset, (int)a_stride);
                            _tile_loadd(3, b_tiles + (b_n1_base + k_tile) * tile_byte_size, 64);

                            // Compute: A1 × B0 → C[1,0], A0 × B1 → C[0,1], A1 × B1 → C[1,1]
                            _tile_dpbssd(6, 1, 2);
                            _tile_dpbssd(5, 0, 3);
                            _tile_dpbssd(7, 1, 3);

                            // Load next iteration: A0, B0
                            _tile_loadd(0, a_row0 + next_k_offset, (int)a_stride);
                            _tile_loadd(2, b_tiles + (b_n0_base + k_tile + 1) * tile_byte_size, 64);
                        }

                        // Epilogue: Process last K tile
                        simsimd_size_t const last_k = num_k_tiles - 1;
                        simsimd_size_t const last_k_offset = last_k * tile_k_elements;

                        _tile_dpbssd(4, 0, 2);
                        _tile_loadd(1, a_row1 + last_k_offset, (int)a_stride);
                        _tile_loadd(3, b_tiles + (b_n1_base + last_k) * tile_byte_size, 64);
                        _tile_dpbssd(6, 1, 2);
                        _tile_dpbssd(5, 0, 3);
                        _tile_dpbssd(7, 1, 3);
                    }
                    else {
                        // Single K tile: no pipelining needed
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(1, a_row1, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_byte_size, 64);
                        _tile_loadd(3, b_tiles + b_n1_base * tile_byte_size, 64);

                        _tile_dpbssd(4, 0, 2);
                        _tile_dpbssd(5, 0, 3);
                        _tile_dpbssd(6, 1, 2);
                        _tile_dpbssd(7, 1, 3);
                    }

                    // Store C tiles directly (aligned path)
                    simsimd_i32_t *c_block = c + m_row * c_stride_i32 + n_col;
                    _tile_stored(4, c_block, (int)c_stride);
                    _tile_stored(5, c_block + 16, (int)c_stride);
                    _tile_stored(6, c_block + 16 * c_stride_i32, (int)c_stride);
                    _tile_stored(7, c_block + 16 * c_stride_i32 + 16, (int)c_stride);
                }
            }
        }

        _tile_release();
    }

    // AMX: M edge rows for full N tiles (rows that don't fill a complete 32-row block)
    if (m > num_m_blocks * 32 && num_n_tiles > 0) {
        simsimd_size_t const m_edge_start = num_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
            simsimd_size_t const n_col = n_tile * 16;

            _tile_zero(4);
            _tile_zero(6);

            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

            simsimd_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            simsimd_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (simsimd_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                simsimd_size_t const k_offset = k_tile * tile_k_elements;
                simsimd_size_t const k_valid = (k_offset + tile_k_elements <= k) ? tile_k_elements : (k - k_offset);

                _simsimd_load_a_tile_i8_masked(a + m_edge_start * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                               (simsimd_i8_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_i8_masked(a + (m_edge_start + 16) * a_stride + k_offset, a_stride, rows_lower,
                                                   k_valid, (simsimd_i8_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // Linear B tile index
                simsimd_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                _tile_loadd(2, b_tiles + b_tile_idx * tile_byte_size, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            SIMSIMD_ALIGN64 simsimd_i32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile, c + m_edge_start * c_stride_i32 + n_col,
                                             c_stride_i32, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + (m_edge_start + 16) * c_stride_i32 + n_col, c_stride_i32,
                                                 rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge columns (columns that don't fill a complete 16-column tile)
    if (n_edge_cols > 0 && n_edge_tiles != NULL) {
        _simsimd_dots_i8i8i32_avx512_edge(a, n_edge_tiles, c + full_n_cols, m, n_edge_cols, k, a_stride, k,
                                          c_stride_i32);
    }

    // AVX-512: M edge × N edge corner
    if (m > num_m_blocks * 32 && n_edge_cols > 0 && n_edge_tiles != NULL) {
        simsimd_size_t const m_edge_start = num_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_dots_i8i8i32_avx512_edge(a + m_edge_start * a_stride, n_edge_tiles,
                                          c + m_edge_start * c_stride_i32 + full_n_cols, m_edge_rows, n_edge_cols, k,
                                          a_stride, k, c_stride_i32);
    }
}

/*  I8 → I32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
SIMSIMD_INTERNAL void _simsimd_dots_i8i8i32_sapphire_misaligned(   //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    simsimd_dots_packed_header_t const *header = (simsimd_dots_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;
    simsimd_size_t const n_edge_offset = header->n_edge_offset;

    simsimd_size_t const tile_cols_i8 = 64;
    simsimd_size_t const tile_elements_i8 = 1024;

    simsimd_i8_t const *tiles_ptr = (simsimd_i8_t const *)((char const *)b_packed + 64);
    simsimd_i8_t const *n_edge_ptr = (n_edge_offset > 0)
                                         ? (simsimd_i8_t const *)((char const *)b_packed + n_edge_offset)
                                         : NULL;

    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_i32_t);
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = n / 32;
    simsimd_size_t const full_n = full_n_tiles * 16;
    simsimd_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // Stack buffers for tile I/O
    SIMSIMD_ALIGN64 simsimd_i8_t a_buf_upper[16][64];
    SIMSIMD_ALIGN64 simsimd_i8_t a_buf_lower[16][64];
    SIMSIMD_ALIGN64 simsimd_i32_t c_buf[16][16];

    // AMX: Full 32×32 blocks through buffers
    if (full_m_blocks > 0 && full_n_blocks > 0 && full_k_tiles > 0) {
        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
            simsimd_size_t const row_block = bi * 32;

            for (simsimd_size_t bj = 0; bj < full_n_blocks; bj++) {
                simsimd_size_t const col_block = bj * 32;

                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);

                simsimd_size_t const b_tile_n0 = bj * 2;
                simsimd_size_t const b_tile_n1 = bj * 2 + 1;

                for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                    simsimd_size_t const k_offset = bk * 64;

                    // Load A through buffers using AVX-512
                    for (simsimd_size_t r = 0; r < 16; r++) {
                        simsimd_i8_t const *src_upper = a + (row_block + r) * a_stride + k_offset;
                        simsimd_i8_t const *src_lower = a + (row_block + 16 + r) * a_stride + k_offset;
                        __m512i upper_row = _mm512_loadu_si512((__m512i const *)src_upper);
                        __m512i lower_row = _mm512_loadu_si512((__m512i const *)src_lower);
                        _mm512_store_si512((__m512i *)a_buf_upper[r], upper_row);
                        _mm512_store_si512((__m512i *)a_buf_lower[r], lower_row);
                    }
                    __asm__ volatile("" ::: "memory");

                    _tile_loadd(0, a_buf_upper, 64);
                    _tile_loadd(1, a_buf_lower, 64);

                    simsimd_size_t morton_idx0 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0,
                                                                                     (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                    simsimd_size_t morton_idx1 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1,
                                                                                     (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    simsimd_i8_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_i8;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }

                // Store C through buffers using AVX-512
                _tile_stored(4, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + r) * c_stride_elements + col_block), row);
                }

                _tile_stored(5, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + r) * c_stride_elements + col_block + 16), row);
                }

                _tile_stored(6, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + 16 + r) * c_stride_elements + col_block), row);
                }

                _tile_stored(7, c_buf, 64);
                for (simsimd_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + 16 + r) * c_stride_elements + col_block + 16),
                                        row);
                }
            }
        }

        _tile_release();
    }

    // AMX: M edge rows for full N tiles (through buffers)
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t tj = 0; tj < full_n_tiles; tj++) {
            simsimd_size_t const col_block = tj * 16;
            simsimd_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            // Zero buffers for edge
            for (simsimd_size_t r = 0; r < 16; r++) {
                _mm512_store_si512((__m512i *)a_buf_upper[r], _mm512_setzero_si512());
                _mm512_store_si512((__m512i *)a_buf_lower[r], _mm512_setzero_si512());
            }

            simsimd_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            simsimd_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                simsimd_size_t const k_offset = bk * tile_cols_i8;
                simsimd_size_t const k_valid = (k_offset + tile_cols_i8 <= k) ? tile_cols_i8 : (k - k_offset);

                _simsimd_load_a_tile_i8_masked(a + m_edge_start * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                               (simsimd_i8_t *)a_buf_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_i8_masked(a + (m_edge_start + 16) * a_stride + k_offset, a_stride, rows_lower,
                                                   k_valid, (simsimd_i8_t *)a_buf_lower);
                }

                _tile_loadd(0, a_buf_upper, 64);
                _tile_loadd(1, a_buf_lower, 64);

                simsimd_size_t morton_idx0 = _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0,
                                                                                 (simsimd_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                simsimd_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_buf, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_buf, c + m_edge_start * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_buf, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_buf,
                                                 c + (m_edge_start + 16) * c_stride_elements + col_block,
                                                 c_stride_elements, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0 && n_edge_ptr != NULL) {
        _simsimd_dots_i8i8i32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride, k, c_stride_elements);
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0 && n_edge_ptr != NULL) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_dots_i8i8i32_avx512_edge(a + m_edge_start * a_stride, n_edge_ptr,
                                          c + m_edge_start * c_stride_elements + full_n, m_edge_rows, n_edge_rows, k,
                                          a_stride, k, c_stride_elements);
    }
}

/*  I8 → I32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_sapphire_amx(             //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes for I8 A tile row and I32 C tile row)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) _simsimd_dots_i8i8i32_sapphire_aligned(a, b_packed, c, m, n, k, a_stride, c_stride);
    else
        _simsimd_dots_i8i8i32_sapphire_misaligned(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 * rsqrt(a_norm[i] * b_norm[j])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = n * sizeof(i8).
 */
SIMSIMD_PUBLIC void simsimd_dots_i8i8i8_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,      //
    simsimd_size_t c_stride,                          //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms) {

    simsimd_size_t const c_stride_i32 = c_stride / sizeof(simsimd_i32_t);
    simsimd_i32_t const *c_i32 = (simsimd_i32_t const *)c;
    simsimd_i8_t *c_i8 = (simsimd_i8_t *)c;

    // Use space after I8 output for precomputed b_rsqrt (I8 output is 4x smaller than I32 input)
    simsimd_f32_t *b_rsqrt = (simsimd_f32_t *)(c_i8 + m * n);

    // Precompute rsqrt of all b_norms using AVX512 (16 at a time)
    __m512 half_vec = _mm512_set1_ps(0.5f);
    __m512 three_halves_vec = _mm512_set1_ps(1.5f);
    simsimd_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        __m512i b_norms_i32 = _mm512_loadu_si512(b_squared_norms + j);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        // Newton-Raphson refinement
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        // Zero out rsqrt where norm was zero
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask, rsqrt_vec);
        _mm512_storeu_ps(b_rsqrt + j, rsqrt_vec);
    }

    // Handle remaining b_norms with masked operations
    if (j < n) {
        __mmask16 tail_mask = (__mmask16)((1u << (n - j)) - 1);
        __m512i b_norms_i32 = _mm512_maskz_loadu_epi32(tail_mask, b_squared_norms + j);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask & tail_mask, rsqrt_vec);
        _mm512_mask_storeu_ps(b_rsqrt + j, tail_mask, rsqrt_vec);
    }

    __m512 scale_vec = _mm512_set1_ps(127.0f);

    for (simsimd_size_t row = 0; row < m; row++) {
        simsimd_i32_t const *src_row = c_i32 + row * c_stride_i32;
        simsimd_i8_t *dst_row = c_i8 + row * n;

        // Compute rsqrt of a_norm for this row, broadcast to vector
        simsimd_f32_t a_norm_f32 = (simsimd_f32_t)a_squared_norms[row];
        simsimd_f32_t a_rsqrt_val = 0.0f;
        if (a_norm_f32 > 0.0f) {
            __m128 a_vec = _mm_set_ss(a_norm_f32);
            __m128 rsqrt_s = _mm_rsqrt_ss(a_vec);
            rsqrt_s = _mm_mul_ss(
                rsqrt_s, _mm_sub_ss(_mm_set_ss(1.5f),
                                    _mm_mul_ss(_mm_set_ss(0.5f), _mm_mul_ss(a_vec, _mm_mul_ss(rsqrt_s, rsqrt_s)))));
            a_rsqrt_val = _mm_cvtss_f32(rsqrt_s);
        }
        __m512 a_rsqrt_vec = _mm512_set1_ps(a_rsqrt_val);
        __m512 row_scale = _mm512_mul_ps(a_rsqrt_vec, scale_vec);

        simsimd_size_t col = 0;

        // Process 16 elements at a time
        for (; col + 16 <= n; col += 16) {
            __m512i c_vals = _mm512_loadu_si512(src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_loadu_ps(b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            // Saturating pack I32 → I8 (16 values → 16 bytes in low 128 bits)
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i *)(dst_row + col), result_i8);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512i c_vals = _mm512_maskz_loadu_epi32(tail_mask, src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_maskz_loadu_ps(tail_mask, b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_mask_storeu_epi8(dst_row + col, tail_mask, result_i8);
        }
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE_AMX

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), apply_to = function)

// I8 GEMM packing functions (interleave=4 for VPDPBUSD/VNNI, k_align=8)
SIMSIMD_MAKE_DOTS_PACKED_SIZE(ice, i8, i32, 8)
SIMSIMD_MAKE_DOTS_PACK(ice, i8, i32, 4, 8)

// I8 GEMM: MR=8, k_tile=8, nr_acc=4 for 4× throughput
SIMSIMD_MAKE_DOTS(i8i8i32_ice, i8, i32, simsimd_dot_outer_i8x8_state_1x16_ice_t, simsimd_dot_outer_i8x8_init_1x16_ice,
                  simsimd_dot_outer_i8x8_update_1x16_ice, simsimd_dot_outer_i8x8_finalize_1x16_ice,
                  /*k_tile=*/8, /*MR=*/8, /*nr_acc=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_bf16bf16f32_packed_size(simsimd_size_t n, simsimd_size_t k) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    return simsimd_dots_bf16bf16f32_packed_size_sapphire_amx(n, k);
#elif SIMSIMD_TARGET_GENOA
    return simsimd_dots_bf16bf16f32_packed_size_genoa(n, k);
#else
    return simsimd_dots_bf16bf16f32_packed_size_serial(n, k);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32_pack(simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k,
                                                  simsimd_size_t b_stride, void *b_packed) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_bf16bf16f32_pack_sapphire_amx(b, n, k, b_stride, b_packed);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_bf16bf16f32_pack_genoa(b, n, k, b_stride, b_packed);
#else
    simsimd_dots_bf16bf16f32_pack_serial(b, n, k, b_stride, b_packed);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_bf16bf16f32(simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c,
                                             simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,
                                             simsimd_size_t a_stride, simsimd_size_t c_stride) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_bf16bf16f32_sapphire_amx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_bf16bf16f32_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    simsimd_dots_bf16bf16f32_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_bf16bf16bf16(void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_bf16bf16bf16_sapphire_amx(c, m, n, c_stride);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_bf16bf16bf16_genoa(c, m, n, c_stride);
#else
    simsimd_dots_bf16bf16bf16_serial(c, m, n, c_stride);
#endif
}

SIMSIMD_PUBLIC simsimd_size_t simsimd_dots_i8i8i32_packed_size(simsimd_size_t n, simsimd_size_t k) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    return simsimd_dots_i8i8i32_packed_size_sapphire_amx(n, k);
#elif SIMSIMD_TARGET_GENOA
    return simsimd_dots_i8i8i32_packed_size_genoa(n, k);
#else
    return simsimd_dots_i8i8i32_packed_size_serial(n, k);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_i8i8i32_pack(simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k,
                                              simsimd_size_t b_stride, void *b_packed) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_i8i8i32_pack_sapphire_amx(b, n, k, b_stride, b_packed);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_i8i8i32_pack_genoa(b, n, k, b_stride, b_packed);
#else
    simsimd_dots_i8i8i32_pack_serial(b, n, k, b_stride, b_packed);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_i8i8i32(simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c,
                                         simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride,
                                         simsimd_size_t c_stride) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_i8i8i32_sapphire_amx(a, b_packed, c, m, n, k, a_stride, c_stride);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_i8i8i32_genoa(a, b_packed, c, m, n, k, a_stride, c_stride);
#else
    simsimd_dots_i8i8i32_serial(a, b_packed, c, m, n, k, a_stride, c_stride);
#endif
}

SIMSIMD_PUBLIC void simsimd_dots_i8i8i8(void *c, simsimd_size_t m, simsimd_size_t n, simsimd_size_t c_stride,
                                        simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms) {
#if SIMSIMD_TARGET_SAPPHIRE_AMX
    simsimd_dots_i8i8i8_sapphire_amx(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dots_i8i8i8_genoa(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#else
    simsimd_dots_i8i8i8_serial(c, m, n, c_stride, a_squared_norms, b_squared_norms);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
