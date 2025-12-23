/**
 *  @file       matmul.h
 *  @brief      SIMD-accelerated Matrix Multiplication for low-precision numerics.
 *  @author     Ash Vardanian
 *  @date       September 14, 2024
 *
 *  Implements matrix-multiplication kernels, focusing on mixed precision, computing:
 *  C[m×n] = A[m×k] × B[n×k]ᵀ with row-major inputs for Neural Network inference and beyond:
 *
 *  - k-NN search: Euclidean distances via ||a-b||² = ||a||² + ||b||² - 2(a·b)
 *  - Cosine similarity matrices: (a·b) / (||a||·||b||)
 *  - k-means clustering, DBSCAN, hierarchical clustering
 *  - Video frame flow estimation, motion detection
 *  - DSP filter banks, radar pulse integration
 *
 *  They are heavily tuned for SIMD-divisible sizes, often using a separate suboptimal code path
 *  for non-divisible sub-optimal input sizes. Bad for performance, but kees the code maintainable.
 *
 *  For datatypes:
 *  - 16-bit brain floats (BF16) accumulating into 32-bit floats
 *  - 16-bit brain floats (BF16) accumulating into 32-bit floats with truncation to BF16
 *  - 8-bit signed integers (I8) accumulating into 32-bit integers
 *  - 8-bit signed integers (I8) accumulating into 32-bit integers with re-normalization to I8
 *
 *  For hardware architectures:
 *  - x86: Haswell (AVX2), Genoa (AVX512-BF16), Sapphire Rapids (AMX)
 *  - Arm: NEON, SVE, SME
 *
 *  @section    Memory Layout and Transpose Semantics
 *
 *  All matrices use row-major storage. Column-major is NOT supported.
 *  The kernel computes C = A × Bᵀ where:
 *  - A is (m × k): m rows, k columns, stride = a_stride bytes between rows
 *  - B is (n × k): n rows, k columns, stride = b_stride bytes between rows
 *  - C is (m × n): m rows, n columns, stride = c_stride bytes between rows
 *
 *  This means C[i,j] = dot(row i of A, row j of B) = Σₗ A[i,l] × B[j,l].
 *
 *  To compute standard A × B (where B is k × n), pass Bᵀ to the packing function:
 *
 *      // Standard matmul: C[m×n] = A[m×k] × B[k×n]
 *      // B is stored row-major as k rows of n elements
 *      // Treat it as Bᵀ: n rows of k elements with stride = sizeof(element)
 *      simsimd_matmul_bf16_pack_sapphire_amx(b, n, k, sizeof(simsimd_bf16_t), b_packed);
 *      simsimd_matmul_bf16_f32_sapphire_amx(a, b_packed, c, m, n, k, a_stride, c_stride);
 *      // Result: C = A × (Bᵀ)ᵀ = A × B
 *
 *  @section    Two-Phase API for Static Weights
 *
 *  Matrix multiplication hardware (AMX, SME) requires specific data layouts that differ
 *  from standard row-major ordering. Since one matrix (typically weights in neural networks)
 *  is often static, we provide a two-phase API: pack once, multiply many times.
 *
 *      // Similarity search: C[m×n] = queries[m×k] × database[n×k]ᵀ
 *      // Both matrices stored row-major, each row is one vector of dimension k
 *      simsimd_size_t packed_bytes = simsimd_matmul_bf16_packed_size_sapphire_amx(n, k);
 *      void *b_packed = malloc(packed_bytes);
 *      simsimd_matmul_bf16_pack_sapphire_amx(database, n, k, k * sizeof(simsimd_bf16_t), b_packed);
 *      simsimd_matmul_bf16_f32_sapphire_amx(queries, b_packed, c, m, n, k, ...);
 *      // Result: C[i,j] = dot(query i, database vector j)
 *
 *  The packed format is opaque and backend-specific. AMX expects (16×32) tiles with interleaved
 *  pairs, while NEON/SVE use different arrangements optimized for their vector lengths.
 *
 *  @section    Why INT8 and Not UINT8?
 *
 *  Unsigned 8-bit integers were considered but deprioritized. The industry has converged on
 *  signed INT8 as the standard for quantized inference:
 *
 *  | Framework       | Default  | Notes                                    |
 *  |-----------------|----------|------------------------------------------|
 *  | PyTorch         | qint8    | New X86 backend uses INT8 via oneDNN     |
 *  | TensorFlow Lite | int8     | Actively removing UINT8 support          |
 *  | ONNX Runtime    | S8S8     | "Should be the first choice"             |
 *  | TensorRT        | INT8     | Symmetric [-128,127], no UINT8 option    |
 *  | ARM CMSIS-NN    | int8     | Follows TFLite INT8 spec exactly         |
 *
 *  @section    Why No Alpha/Beta Scaling?
 *
 *  BLAS-style `C = α·A·B + β·C` scaling was considered but omitted. While useful for scientific
 *  computing (iterative solvers, matrix factorizations), it's rarely used in ML inference where
 *  frameworks handle such operations via graph fusion. More importantly, on chips with separate
 *  physical registers for vector and matrix operations (like AMX), moving scalars between register
 *  files adds transfer latency that negates any benefit.
 *
 *  @section    Why Not Pad N Dimension to Eliminate Edge Handling?
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
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  Matrix Multiplication in 40 lines: https://en.algorithmica.org/hpc/algorithms/matmul/
 *  LLaMA CPU optimization: https://justine.lol/matmul/
 *  SME outer-product notes: https://github.com/tzakharko/m4-sme-exploration
 */
#ifndef SIMSIMD_MATMUL_H
#define SIMSIMD_MATMUL_H

#include "types.h"

#include "dot.h" // `_simsimd_bf16x16_to_f32x16_skylake`

#ifdef __cplusplus
extern "C" {
#endif

/*  Tunable tile sizes for cache blocking.
 *  These can be overridden before including this header to tune for specific cache sizes.
 *  The L1 tile should fit 3 matrices (A, B, C) in L1 cache with room for accumulators.
 */
#ifndef SIMSIMD_MATMUL_L1_TILE_M
#define SIMSIMD_MATMUL_L1_TILE_M 128
#endif
#ifndef SIMSIMD_MATMUL_L1_TILE_N
#define SIMSIMD_MATMUL_L1_TILE_N 128
#endif
#ifndef SIMSIMD_MATMUL_L1_TILE_K
#define SIMSIMD_MATMUL_L1_TILE_K 128
#endif

/*  Serial backends for packing and multiplication.
 *  These are portable reference implementations with no SIMD dependencies.
 *  Serial packing simply copies B transposed - no special layout required.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_serial(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_serial(             //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_serial(                  //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_serial( //
    void *c, simsimd_size_t m, simsimd_size_t n,        //
    simsimd_size_t c_stride);

SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_serial(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_serial(             //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_serial(                  //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_serial( //
    void *c, simsimd_size_t m, simsimd_size_t n,      //
    simsimd_size_t c_stride,                          //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms);

/*  Ice Lake backends using AVX-512 with BF16 extensions.
 *  Packing interleaves elements for efficient SIMD broadcast patterns.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_genoa(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_genoa(              //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_genoa(                   //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_genoa( //
    void *c, simsimd_size_t m, simsimd_size_t n,     //
    simsimd_size_t c_stride,                         //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products and VPDPBUSD for INT8.
 *  Packing interleaves elements for efficient SIMD broadcast patterns.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_genoa(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_genoa(              //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_genoa(                   //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_genoa( //
    void *c, simsimd_size_t m, simsimd_size_t n,       //
    simsimd_size_t c_stride);

#if SIMSIMD_TARGET_SAPPHIRE_AMX
/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16×32) BF16 or (16×64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_sapphire_amx(       //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_sapphire_amx(            //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,              //
    simsimd_size_t c_stride);

SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_sapphire_amx(       //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_sapphire_amx(            //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,            //
    simsimd_size_t c_stride,                                //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms);
#endif // SIMSIMD_TARGET_SAPPHIRE_AMX (declarations)

/*  Legacy unpacked matmul implementations for direct A×B^T operations.
 *  These operate on unpacked matrices directly (no two-phase pack/multiply API).
 *  Use when B matrix is not reused across multiple multiplications.
 *
 *  Naming: simsimd_matmul_<type>_<variant>_unpacked
 *  - serial: basic tiled implementation
 *  - accurate: higher precision accumulator
 */
#define SIMSIMD_MAKE_MATMUL_UNPACKED(name, input_type, accumulator_type, output_type, load_and_convert,       \
                                     convert_and_store)                                                       \
    SIMSIMD_PUBLIC void simsimd_matmul_##input_type##_##name##_unpacked(                                      \
        simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols, simsimd_##input_type##_t const *a, \
        simsimd_size_t a_stride, simsimd_##input_type##_t const *b, simsimd_size_t b_stride,                  \
        simsimd_##output_type##_t *c, simsimd_size_t c_stride) {                                              \
        for (simsimd_size_t i = 0; i < a_rows; ++i) {                                                         \
            simsimd_##input_type##_t const *a_row =                                                           \
                (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)a, i * a_stride);         \
            simsimd_##output_type##_t *c_row =                                                                \
                (simsimd_##output_type##_t *)_simsimd_advance_by_bytes((void *)c, i * c_stride);              \
            for (simsimd_size_t j = 0; j < b_rows; ++j) {                                                     \
                simsimd_##input_type##_t const *b_row =                                                       \
                    (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)b, j * b_stride);     \
                simsimd_##accumulator_type##_t sum = 0;                                                       \
                for (simsimd_size_t k = 0; k < cols; ++k) {                                                   \
                    simsimd_##accumulator_type##_t aik = load_and_convert(a_row + k);                         \
                    simsimd_##accumulator_type##_t bjk = load_and_convert(b_row + k);                         \
                    sum += aik * bjk;                                                                         \
                }                                                                                             \
                convert_and_store(sum, c_row + j);                                                            \
            }                                                                                                 \
        }                                                                                                     \
    }

#define SIMSIMD_MAKE_TILED_UNPACKED(name, input_type, accumulator_type, output_type, load_and_convert,                \
                                    convert_and_store, tile_size)                                                     \
    SIMSIMD_PUBLIC void simsimd_matmul_##input_type##_##name##_unpacked(                                              \
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
                        simsimd_##output_type##_t *c_row =                                                            \
                            (simsimd_##output_type##_t *)_simsimd_advance_by_bytes((void *)c, i * c_stride);          \
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

// clang-format off
SIMSIMD_MAKE_TILED_UNPACKED(serial, f64, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f64_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, f32, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f32_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, f16, f32, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)     // simsimd_matmul_f16_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, bf16, f32, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16, 16) // simsimd_matmul_bf16_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(serial, i8, i64, i8, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)          // simsimd_matmul_i8_serial_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, f32, f64, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)      // simsimd_matmul_f32_accurate_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, f16, f64, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)   // simsimd_matmul_f16_accurate_unpacked
SIMSIMD_MAKE_TILED_UNPACKED(accurate, bf16, f64, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16, 16) // simsimd_matmul_bf16_accurate_unpacked
// clang-format on

/*  Serial compact functions: simple scalar implementations for post-matmul conversion.
 *  These work on any platform without SIMD requirements.
 */

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride, writes BF16 tightly packed (stride = n * sizeof(bf16)).
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_serial( //
    void *c, simsimd_size_t m, simsimd_size_t n,        //
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
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_serial( //
    void *c, simsimd_size_t m, simsimd_size_t n,      //
    simsimd_size_t c_stride,                          //
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

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), apply_to = function)

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
 *  Call `simsimd_enable_capabilities(simsimd_cap_sapphire_amx_k)` once per thread
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
SIMSIMD_INTERNAL void _simsimd_matmul_bf16_f32_avx512_edge(                  //
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
SIMSIMD_INTERNAL void _simsimd_matmul_i8_i32_avx512_edge(                //
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

/*  Packed buffer header for hybrid layout (64-byte aligned).
 *  Layout: [header][full tiles Morton-ordered][k_edge row-major][n_edge row-major]
 *
 *  Full tiles: Interior tiles that are complete (no partial rows/cols)
 *  K edge: Remaining columns (k % tile_cols != 0) stored in row-major for easy AVX-512 access
 *  N edge: Remaining rows (n % tile_rows != 0) stored in row-major for easy AVX-512 access
 */
typedef struct {
    simsimd_u32_t full_n_tiles;  // Number of full N tiles (16 rows each)
    simsimd_u32_t full_k_tiles;  // Number of full K tiles (32/64 cols each)
    simsimd_u32_t k_edge_cols;   // Remaining K columns (0 to tile_cols-1)
    simsimd_u32_t n_edge_rows;   // Remaining N rows (0 to 15)
    simsimd_u32_t k_edge_offset; // Byte offset to K edge region
    simsimd_u32_t n_edge_offset; // Byte offset to N edge region
    simsimd_u32_t n_total;       // Original N dimension
    simsimd_u32_t k_total;       // Original K dimension
    simsimd_u32_t reserved[8];   // Padding to 64 bytes
} simsimd_matmul_packed_header_t;

/*  BF16 packed buffer size: header + all tiles for full N rows + N edge.
 *  Hybrid layout:
 *    - Tiles include K remainder (zero-padded) for AMX to handle full dot products
 *    - N edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 32;
    simsimd_size_t const tile_bytes = 512 * sizeof(simsimd_bf16_t); // 16×32×2 = 1KB

    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    simsimd_size_t size = sizeof(simsimd_matmul_packed_header_t);

    // All tiles for full N rows (Morton-ordered, pair-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(simsimd_bf16_t);

    return size;
}

/*  I8 packed buffer size: header + all tiles for full N rows + N edge.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_sapphire_amx(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 64;
    simsimd_size_t const tile_bytes = 1024 * sizeof(simsimd_i8_t); // 16×64×1 = 1KB

    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    simsimd_size_t size = sizeof(simsimd_matmul_packed_header_t);

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
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_sapphire_amx(       //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 32;
    simsimd_size_t const tile_elements = 512;
    simsimd_size_t const tile_bytes = tile_elements * sizeof(simsimd_bf16_t);
    simsimd_size_t const b_stride_elements = b_stride / sizeof(simsimd_bf16_t);

    // Compute layout dimensions
    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Includes K remainder tile
    simsimd_size_t const k_remainder = k - (tiles_along_k > 0 ? (tiles_along_k - 1) * tile_cols : 0);
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;
    simsimd_size_t const total_tiles = full_n_tiles * tiles_along_k;

    // Write header
    simsimd_matmul_packed_header_t *header = (simsimd_matmul_packed_header_t *)b_packed;
    header->full_n_tiles = (simsimd_u32_t)full_n_tiles;
    header->full_k_tiles = (simsimd_u32_t)tiles_along_k; // Now includes K remainder
    header->k_edge_cols = 0;                             // No separate K edge (included in tiles)
    header->n_edge_rows = (simsimd_u32_t)n_edge_rows;
    header->n_total = (simsimd_u32_t)n;
    header->k_total = (simsimd_u32_t)k;

    // Compute offsets
    simsimd_size_t tiles_offset = sizeof(simsimd_matmul_packed_header_t);
    simsimd_size_t n_edge_offset = tiles_offset + total_tiles * tile_bytes;

    header->k_edge_offset = 0; // No K edge region
    header->n_edge_offset = (simsimd_u32_t)n_edge_offset;

    // Pointers to regions
    simsimd_bf16_t *tiles_ptr = (simsimd_bf16_t *)((char *)b_packed + tiles_offset);
    simsimd_bf16_t *n_edge_ptr = (simsimd_bf16_t *)((char *)b_packed + n_edge_offset);

    // Zero all tiles region (handles K remainder padding automatically)
    for (simsimd_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack all tiles for full N rows with Morton ordering and pair-interleaving
    for (simsimd_size_t tile_n = 0; tile_n < full_n_tiles; tile_n++) {
        for (simsimd_size_t tile_k = 0; tile_k < tiles_along_k; tile_k++) {
            // Morton Z-curve tile index
            simsimd_size_t morton_idx =
                _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)tile_n, (simsimd_u32_t)tile_k);
            simsimd_size_t tile_index = (morton_idx < total_tiles) ? morton_idx : (tile_n * tiles_along_k + tile_k);
            simsimd_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            simsimd_size_t const row_start = tile_n * tile_rows;
            simsimd_size_t const col_start = tile_k * tile_cols;
            simsimd_size_t const cols_in_tile =
                (tile_k == tiles_along_k - 1 && k_remainder > 0 && k_remainder < tile_cols) ? (k - col_start)
                                                                                            : tile_cols;

            // Pack with pair-interleaving (required by TDPBF16PS)
            for (simsimd_size_t local_row = 0; local_row < tile_rows; local_row++) {
                for (simsimd_size_t local_col = 0; local_col < cols_in_tile; local_col++) {
                    simsimd_size_t source_idx = (row_start + local_row) * b_stride_elements + (col_start + local_col);
                    simsimd_size_t packed_idx = (local_col / 2) * 32 + local_row * 2 + (local_col % 2);
                    tile_output[packed_idx] = b[source_idx];
                }
            }
            // Remaining columns (if any) stay zero from initialization
        }
    }

    // Pack N edge rows (row-major) - all K columns
    if (n_edge_rows > 0) {
        simsimd_size_t const n_edge_start = full_n_tiles * tile_rows;
        for (simsimd_size_t row = 0; row < n_edge_rows; row++) {
            for (simsimd_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(n_edge_start + row) * b_stride_elements + col];
            }
        }
    }
}

/*  Pack I8 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: Morton Z-curve ordered, quad-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_sapphire_amx(       //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 64;
    simsimd_size_t const tile_elements = 1024;
    simsimd_size_t const tile_bytes = tile_elements * sizeof(simsimd_i8_t);

    // Compute layout dimensions
    simsimd_size_t const full_n_tiles = n / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Includes K remainder tile
    simsimd_size_t const k_remainder = k - (tiles_along_k > 0 ? (tiles_along_k - 1) * tile_cols : 0);
    simsimd_size_t const n_edge_rows = n - full_n_tiles * tile_rows;
    simsimd_size_t const total_tiles = full_n_tiles * tiles_along_k;

    // Write header
    simsimd_matmul_packed_header_t *header = (simsimd_matmul_packed_header_t *)b_packed;
    header->full_n_tiles = (simsimd_u32_t)full_n_tiles;
    header->full_k_tiles = (simsimd_u32_t)tiles_along_k; // Now includes K remainder
    header->k_edge_cols = 0;                             // No separate K edge (included in tiles)
    header->n_edge_rows = (simsimd_u32_t)n_edge_rows;
    header->n_total = (simsimd_u32_t)n;
    header->k_total = (simsimd_u32_t)k;

    // Compute offsets
    simsimd_size_t tiles_offset = sizeof(simsimd_matmul_packed_header_t);
    simsimd_size_t n_edge_offset = tiles_offset + total_tiles * tile_bytes;

    header->k_edge_offset = 0; // No K edge region
    header->n_edge_offset = (simsimd_u32_t)n_edge_offset;

    // Pointers to regions
    simsimd_i8_t *tiles_ptr = (simsimd_i8_t *)((char *)b_packed + tiles_offset);
    simsimd_i8_t *n_edge_ptr = (simsimd_i8_t *)((char *)b_packed + n_edge_offset);

    // Zero all tiles region (handles K remainder padding automatically)
    for (simsimd_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack all tiles for full N rows with Morton ordering and quad-interleaving
    for (simsimd_size_t tile_n = 0; tile_n < full_n_tiles; tile_n++) {
        for (simsimd_size_t tile_k = 0; tile_k < tiles_along_k; tile_k++) {
            // Morton Z-curve tile index
            simsimd_size_t morton_idx =
                _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)tile_n, (simsimd_u32_t)tile_k);
            simsimd_size_t tile_index = (morton_idx < total_tiles) ? morton_idx : (tile_n * tiles_along_k + tile_k);
            simsimd_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            simsimd_size_t const row_start = tile_n * tile_rows;
            simsimd_size_t const col_start = tile_k * tile_cols;
            simsimd_size_t const cols_in_tile =
                (tile_k == tiles_along_k - 1 && k_remainder > 0 && k_remainder < tile_cols) ? (k - col_start)
                                                                                            : tile_cols;

            // Pack with quad-interleaving (required by TDPBSSD)
            for (simsimd_size_t local_row = 0; local_row < tile_rows; local_row++) {
                for (simsimd_size_t local_col = 0; local_col < cols_in_tile; local_col++) {
                    simsimd_size_t source_idx = (row_start + local_row) * b_stride + (col_start + local_col);
                    simsimd_size_t packed_idx = (local_col / 4) * 64 + local_row * 4 + (local_col % 4);
                    tile_output[packed_idx] = b[source_idx];
                }
            }
            // Remaining columns (if any) stay zero from initialization
        }
    }

    // Pack N edge rows (row-major) - all K columns
    if (n_edge_rows > 0) {
        simsimd_size_t const n_edge_start = full_n_tiles * tile_rows;
        for (simsimd_size_t row = 0; row < n_edge_rows; row++) {
            for (simsimd_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(n_edge_start + row) * b_stride + col];
            }
        }
    }
}

/*  BF16 → F32 matmul (aligned path): Direct tile loads/stores when stride >= 64 bytes.
 *  This is the fast path - no intermediate buffers for A or C.
 */
SIMSIMD_INTERNAL void _simsimd_matmul_bf16_f32_sapphire_aligned(     //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Read header for hybrid layout
    simsimd_matmul_packed_header_t const *header = (simsimd_matmul_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;

    simsimd_bf16_t const *tiles_ptr =
        (simsimd_bf16_t const *)((char const *)b_packed + sizeof(simsimd_matmul_packed_header_t));
    simsimd_bf16_t const *n_edge_ptr = (simsimd_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    simsimd_size_t const tile_cols_bf16 = 32;
    simsimd_size_t const tile_elements_bf16 = 512;

    simsimd_size_t const a_stride_elements = a_stride / sizeof(simsimd_bf16_t);
    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_f32_t);
    simsimd_size_t const full_n = full_n_tiles * 16;
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = full_n_tiles / 2;
    simsimd_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // AMX: Full 32×32 blocks with direct I/O
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

                    // Direct A tile loads
                    _tile_loadd(0, a + row_block * a_stride_elements + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride_elements + k_offset, (int)a_stride);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    simsimd_bf16_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_bf16;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }

                // Direct C stores
                _tile_stored(4, c + row_block * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(5, c + row_block * c_stride_elements + col_block + 16, (int)c_stride);
                _tile_stored(6, c + (row_block + 16) * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(7, c + (row_block + 16) * c_stride_elements + col_block + 16, (int)c_stride);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0) {
        _simsimd_matmul_bf16_f32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride_elements, k,
                                             c_stride_elements);
    }

    // AMX: M edge rows for full N tiles
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t tj = 0; tj < full_n_tiles; tj++) {
            simsimd_size_t const col_block = tj * 16;
            simsimd_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

            simsimd_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            simsimd_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                simsimd_size_t const k_offset = bk * tile_cols_bf16;
                simsimd_size_t const k_valid = (k_offset + tile_cols_bf16 <= k) ? tile_cols_bf16 : (k - k_offset);

                _simsimd_load_a_tile_bf16_masked(a + m_edge_start * a_stride_elements + k_offset, a_stride_elements,
                                                 rows_upper, k_valid, (simsimd_bf16_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_bf16_masked(a + (m_edge_start + 16) * a_stride_elements + k_offset,
                                                     a_stride_elements, rows_lower, k_valid,
                                                     (simsimd_bf16_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                simsimd_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            SIMSIMD_ALIGN64 simsimd_f32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile, c + m_edge_start * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
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

        _simsimd_matmul_bf16_f32_avx512_edge(a + m_edge_start * a_stride_elements, n_edge_ptr,
                                             c + m_edge_start * c_stride_elements + full_n, m_edge_count, n_edge_rows,
                                             k, a_stride_elements, k, c_stride_elements);
    }
}

/*  BF16 → F32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
SIMSIMD_INTERNAL void _simsimd_matmul_bf16_f32_sapphire_misaligned(  //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Read header for hybrid layout
    simsimd_matmul_packed_header_t const *header = (simsimd_matmul_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;

    simsimd_bf16_t const *tiles_ptr =
        (simsimd_bf16_t const *)((char const *)b_packed + sizeof(simsimd_matmul_packed_header_t));
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
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
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
        _simsimd_matmul_bf16_f32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride_elements, k,
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

                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
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

        _simsimd_matmul_bf16_f32_avx512_edge(a + m_edge_start * a_stride_elements, n_edge_ptr,
                                             c + m_edge_start * c_stride_elements + full_n, m_edge_count, n_edge_rows,
                                             k, a_stride_elements, k, c_stride_elements);
    }
}

/*  BF16 → F32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_sapphire_amx(            //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes = 32 BF16 or 16 F32)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) _simsimd_matmul_bf16_f32_sapphire_aligned(a, b_packed, c, m, n, k, a_stride, c_stride);
    else
        _simsimd_matmul_bf16_f32_sapphire_misaligned(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = n * sizeof(bf16).
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,              //
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
 */
SIMSIMD_INTERNAL void _simsimd_matmul_i8_i32_sapphire_aligned(     //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    simsimd_matmul_packed_header_t const *header = (simsimd_matmul_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;
    simsimd_size_t const n_edge_offset = header->n_edge_offset;

    simsimd_size_t const tile_cols_i8 = 64;
    simsimd_size_t const tile_elements_i8 = 1024;

    simsimd_i8_t const *tiles_ptr = (simsimd_i8_t const *)((char const *)b_packed + 64);
    simsimd_i8_t const *n_edge_ptr =
        (n_edge_offset > 0) ? (simsimd_i8_t const *)((char const *)b_packed + n_edge_offset) : NULL;

    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_i32_t);
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = n / 32;
    simsimd_size_t const full_n = full_n_tiles * 16;
    simsimd_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // AMX: Full 32×32 blocks with direct I/O
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

                    // Direct A tile loads
                    _tile_loadd(0, a + row_block * a_stride + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride + k_offset, (int)a_stride);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    simsimd_i8_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_i8;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }

                // Direct C stores
                _tile_stored(4, c + row_block * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(5, c + row_block * c_stride_elements + col_block + 16, (int)c_stride);
                _tile_stored(6, c + (row_block + 16) * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(7, c + (row_block + 16) * c_stride_elements + col_block + 16, (int)c_stride);
            }
        }

        _tile_release();
    }

    // AMX: M edge rows for full N tiles
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_amx_tile_configure_sapphire_amx();

        for (simsimd_size_t tj = 0; tj < full_n_tiles; tj++) {
            simsimd_size_t const col_block = tj * 16;
            simsimd_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

            simsimd_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            simsimd_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (simsimd_size_t bk = 0; bk < full_k_tiles; bk++) {
                simsimd_size_t const k_offset = bk * tile_cols_i8;
                simsimd_size_t const k_valid = (k_offset + tile_cols_i8 <= k) ? tile_cols_i8 : (k - k_offset);

                _simsimd_load_a_tile_i8_masked(a + m_edge_start * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                               (simsimd_i8_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_i8_masked(a + (m_edge_start + 16) * a_stride + k_offset, a_stride, rows_lower,
                                                   k_valid, (simsimd_i8_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                simsimd_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            SIMSIMD_ALIGN64 simsimd_i32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile, c + m_edge_start * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + (m_edge_start + 16) * c_stride_elements + col_block,
                                                 c_stride_elements, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0 && n_edge_ptr != NULL) {
        _simsimd_matmul_i8_i32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride, k,
                                           c_stride_elements);
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0 && n_edge_ptr != NULL) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_matmul_i8_i32_avx512_edge(a + m_edge_start * a_stride, n_edge_ptr,
                                           c + m_edge_start * c_stride_elements + full_n, m_edge_rows, n_edge_rows, k,
                                           a_stride, k, c_stride_elements);
    }
}

/*  I8 → I32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
SIMSIMD_INTERNAL void _simsimd_matmul_i8_i32_sapphire_misaligned(  //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    simsimd_matmul_packed_header_t const *header = (simsimd_matmul_packed_header_t const *)b_packed;
    simsimd_size_t const full_n_tiles = header->full_n_tiles;
    simsimd_size_t const full_k_tiles = header->full_k_tiles;
    simsimd_size_t const n_edge_rows = header->n_edge_rows;
    simsimd_size_t const n_edge_offset = header->n_edge_offset;

    simsimd_size_t const tile_cols_i8 = 64;
    simsimd_size_t const tile_elements_i8 = 1024;

    simsimd_i8_t const *tiles_ptr = (simsimd_i8_t const *)((char const *)b_packed + 64);
    simsimd_i8_t const *n_edge_ptr =
        (n_edge_offset > 0) ? (simsimd_i8_t const *)((char const *)b_packed + n_edge_offset) : NULL;

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

                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    simsimd_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
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

                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire_amx((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
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
        _simsimd_matmul_i8_i32_avx512_edge(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride, k,
                                           c_stride_elements);
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0 && n_edge_ptr != NULL) {
        simsimd_size_t const m_edge_start = full_m_blocks * 32;
        simsimd_size_t const m_edge_rows = m - m_edge_start;

        _simsimd_matmul_i8_i32_avx512_edge(a + m_edge_start * a_stride, n_edge_ptr,
                                           c + m_edge_start * c_stride_elements + full_n, m_edge_rows, n_edge_rows, k,
                                           a_stride, k, c_stride_elements);
    }
}

/*  I8 → I32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_sapphire_amx(            //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes for I8 A tile row and I32 C tile row)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) _simsimd_matmul_i8_i32_sapphire_aligned(a, b_packed, c, m, n, k, a_stride, c_stride);
    else
        _simsimd_matmul_i8_i32_sapphire_misaligned(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 * rsqrt(a_norm[i] * b_norm[j])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = n * sizeof(i8).
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_sapphire_amx( //
    void *c, simsimd_size_t m, simsimd_size_t n,            //
    simsimd_size_t c_stride,                                //
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

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // _SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
