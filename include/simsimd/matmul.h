/**
 *  @file       matmul.h
 *  @brief      SIMD-accelerated Matrix Multiplication for low-precision numerics.
 *  @author     Ash Vardanian
 *  @date       September 14, 2024
 *
 *  Implements matrix-multiplication kernels, focusing on mixed precision,
 *  computing: C[m×n] = A[m×k] × B[n×k]ᵀ with row-major inputs.
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
 *  @section    Two-Phase API for Static Weights
 *
 *  Matrix multiplication hardware (AMX, SME) requires specific data layouts that differ
 *  from standard row-major ordering. Since one matrix (typically weights in neural networks)
 *  is often static, we provide a two-phase API: pack once, multiply many times.
 *
 *      // Pack weights once at model load time
 *      simsimd_size_t packed_bytes = simsimd_matmul_bf16_packed_size_sapphire(n, k);
 *      void *b_packed = malloc(packed_bytes);
 *      simsimd_matmul_bf16_pack_sapphire(b, n, k, b_stride, b_packed);
 *
 *      // Multiply many times during inference
 *      simsimd_matmul_bf16_f32_sapphire(a, b_packed, c, m, n, k, a_stride, c_stride);
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
 *  @section    Applications Beyond Neural Networks
 *
 *  These kernels serve as building blocks for various algorithms:
 *  - k-NN search: Euclidean distances via ||a-b||² = ||a||² + ||b||² - 2(a·b)
 *  - Cosine similarity matrices: (a·b) / (||a||·||b||)
 *  - k-means clustering, DBSCAN, hierarchical clustering
 *  - Video frame flow estimation, motion detection
 *  - DSP filter banks, radar pulse integration
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

#if SIMSIMD_TARGET_SAPPHIRE
/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16×32) BF16 or (16×64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_sapphire(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_sapphire(           //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_sapphire(                //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_sapphire( //
    void *c, simsimd_size_t m, simsimd_size_t n,          //
    simsimd_size_t c_stride);

SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_sapphire(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_sapphire(           //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_sapphire(                //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_sapphire( //
    void *c, simsimd_size_t m, simsimd_size_t n,        //
    simsimd_size_t c_stride,                            //
    simsimd_i32_t const *a_squared_norms, simsimd_i32_t const *b_squared_norms);
#endif // SIMSIMD_TARGET_SAPPHIRE (declarations)

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

#if SIMSIMD_TARGET_SAPPHIRE
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
SIMSIMD_INTERNAL simsimd_u64_t _simsimd_morton_encode_sapphire(simsimd_u32_t tile_row, simsimd_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/*  Configure AMX tile registers.
 *  Called once per kernel invocation (idempotent within a thread).
 *  Sets all 8 tiles to standard 16 rows × 64 bytes layout.
 *
 *  Note: OS permission for AMX must be requested before using AMX instructions.
 *  Call `simsimd_enable_capabilities(simsimd_cap_sapphire_k)` once per thread
 *  before using any Sapphire matmul functions.
 */
SIMSIMD_INTERNAL void _simsimd_amx_tile_configure_sapphire(void) {
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

/*  AVX-512 masked load for A tile (BF16): loads up to 16 rows × 32 cols into aligned buffer.
 *  Uses masked loads to handle edge tiles without element-wise loops.
 *  Note: Memory barrier ensures compiler doesn't optimize away stores before _tile_loadd.
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
    // Memory barrier to ensure stores are visible before _tile_loadd reads
    __asm__ volatile("" ::: "memory");
}

/*  AVX-512 masked load for A tile (I8): loads up to 16 rows × 64 cols into aligned buffer.
 *  Note: Memory barrier ensures compiler doesn't optimize away stores before _tile_loadd.
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
    // Memory barrier to ensure stores are visible before _tile_loadd reads
    __asm__ volatile("" ::: "memory");
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

/*  BF16 packed buffer size: tiles are 16×32 BF16 = 1KB each.
 *  Morton ordering doesn't change the total size, only the layout.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_bf16_packed_size_sapphire(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols_bf16 = 32;
    simsimd_size_t const tile_elements_bf16 = 512;
    simsimd_size_t tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t tiles_along_k = (k + tile_cols_bf16 - 1) / tile_cols_bf16;
    return tiles_along_n * tiles_along_k * tile_elements_bf16 * sizeof(simsimd_bf16_t);
}

/*  I8 packed buffer size: tiles are 16×64 I8 = 1KB each.
 */
SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_sapphire(simsimd_size_t n, simsimd_size_t k) {
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols_i8 = 64;
    simsimd_size_t const tile_elements_i8 = 1024;
    simsimd_size_t tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t tiles_along_k = (k + tile_cols_i8 - 1) / tile_cols_i8;
    return tiles_along_n * tiles_along_k * tile_elements_i8 * sizeof(simsimd_i8_t);
}

/*  Pack BF16 B matrix with Morton Z-curve tile ordering and AMX pair-interleaving.
 *
 *  AMX BF16 tile format: for TDPBF16PS, B tile should have elements arranged so that
 *  consecutive pairs of columns are interleaved by rows:
 *    [col0_row0, col1_row0, col0_row1, col1_row1, ..., col0_row15, col1_row15,
 *     col2_row0, col3_row0, col2_row1, col3_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_pack_sapphire(           //
    simsimd_bf16_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 32;
    simsimd_size_t const tile_elements = 512;

    simsimd_size_t const tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols;
    simsimd_size_t const total_tiles = tiles_along_n * tiles_along_k;
    simsimd_size_t const b_stride_elements = b_stride / sizeof(simsimd_bf16_t);
    simsimd_bf16_t *packed_output = (simsimd_bf16_t *)b_packed;

    // Zero the entire packed buffer first (handles padding for partial tiles)
    for (simsimd_size_t idx = 0; idx < total_tiles * tile_elements; idx++) packed_output[idx] = 0;

    for (simsimd_size_t tile_n = 0; tile_n < tiles_along_n; tile_n++) {
        for (simsimd_size_t tile_k = 0; tile_k < tiles_along_k; tile_k++) {
            // Morton Z-curve tile index for cache-friendly traversal
            simsimd_size_t morton_idx = _simsimd_morton_encode_sapphire((simsimd_u32_t)tile_n, (simsimd_u32_t)tile_k);
            // Clamp to valid range (Morton can produce larger indices than linear)
            simsimd_size_t tile_index = (morton_idx < total_tiles) ? morton_idx : (tile_n * tiles_along_k + tile_k);
            simsimd_bf16_t *tile_output = packed_output + tile_index * tile_elements;

            simsimd_size_t const row_start = tile_n * tile_rows;
            simsimd_size_t const col_start = tile_k * tile_cols;

            // Pack with pair-interleaving (required by TDPBF16PS)
            for (simsimd_size_t local_row = 0; local_row < tile_rows && row_start + local_row < n; local_row++) {
                for (simsimd_size_t local_col = 0; local_col < tile_cols && col_start + local_col < k; local_col++) {
                    simsimd_size_t source_idx = (row_start + local_row) * b_stride_elements + (col_start + local_col);
                    simsimd_size_t packed_idx = (local_col / 2) * 32 + local_row * 2 + (local_col % 2);
                    tile_output[packed_idx] = b[source_idx];
                }
            }
        }
    }
}

/*  Pack I8 B matrix with Morton Z-curve tile ordering and AMX quad-interleaving.
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_sapphire(           //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed) {

    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols = 64;
    simsimd_size_t const tile_elements = 1024;

    simsimd_size_t const tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols;
    simsimd_size_t const total_tiles = tiles_along_n * tiles_along_k;
    simsimd_i8_t *packed_output = (simsimd_i8_t *)b_packed;

    // Zero the entire packed buffer first (handles padding for partial tiles)
    for (simsimd_size_t idx = 0; idx < total_tiles * tile_elements; idx++) packed_output[idx] = 0;

    for (simsimd_size_t tile_n = 0; tile_n < tiles_along_n; tile_n++) {
        for (simsimd_size_t tile_k = 0; tile_k < tiles_along_k; tile_k++) {
            // Morton Z-curve tile index for cache-friendly traversal
            simsimd_size_t morton_idx = _simsimd_morton_encode_sapphire((simsimd_u32_t)tile_n, (simsimd_u32_t)tile_k);
            simsimd_size_t tile_index = (morton_idx < total_tiles) ? morton_idx : (tile_n * tiles_along_k + tile_k);
            simsimd_i8_t *tile_output = packed_output + tile_index * tile_elements;

            simsimd_size_t const row_start = tile_n * tile_rows;
            simsimd_size_t const col_start = tile_k * tile_cols;

            // Pack with quad-interleaving (required by TDPBSSD)
            for (simsimd_size_t local_row = 0; local_row < tile_rows && row_start + local_row < n; local_row++) {
                for (simsimd_size_t local_col = 0; local_col < tile_cols && col_start + local_col < k; local_col++) {
                    simsimd_size_t source_idx = (row_start + local_row) * b_stride + (col_start + local_col);
                    simsimd_size_t packed_idx = (local_col / 4) * 64 + local_row * 4 + (local_col % 4);
                    tile_output[packed_idx] = b[source_idx];
                }
            }
        }
    }
}

/*  BF16 → F32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Uses 2×2 AMX tile pattern (32×32 output blocks):
 *    TMM0, TMM1: A tiles (rows i and i+16)
 *    TMM2, TMM3: B tiles (columns j and j+16)
 *    TMM4-7: C accumulator tiles
 *
 *  Optimized with fast-path/slow-path split:
 *    - Fast path: Full 32×32 blocks with direct tile load/store (no intermediate buffers)
 *    - Slow path: Edge blocks using AVX-512 masked operations
 *
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_f32_sapphire(                //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_f32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,            //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // AMX tile dimensions for BF16
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols_bf16 = 32;
    simsimd_size_t const tile_elements_bf16 = 512;

    _simsimd_amx_tile_configure_sapphire();

    simsimd_size_t const tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols_bf16 - 1) / tile_cols_bf16;
    simsimd_size_t const total_tiles = tiles_along_n * tiles_along_k;
    simsimd_size_t const a_stride_elements = a_stride / sizeof(simsimd_bf16_t);
    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_f32_t);
    simsimd_bf16_t const *b_tiles = (simsimd_bf16_t const *)b_packed;

    // Compute full block counts for fast-path
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = n / 32;
    simsimd_size_t const full_k_blocks = k / 32;
    simsimd_size_t const k_remainder = k - full_k_blocks * 32;

    // Check if strides allow direct tile operations (need 64 bytes = 32 BF16 or 16 F32)
    int const can_direct_load_a = (a_stride >= 64);
    int const can_direct_store_c = (c_stride >= 64);

    // ========== FAST PATH: Full 32×32 blocks ==========
    // No bounds checking needed in inner loops - all tiles are complete
    for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
        simsimd_size_t const row_block = bi * 32;

        for (simsimd_size_t bj = 0; bj < full_n_blocks; bj++) {
            simsimd_size_t const col_block = bj * 32;

            // Zero accumulator tiles
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Get B tile indices for this column block
            simsimd_size_t const b_tile_n0 = bj * 2;
            simsimd_size_t const b_tile_n1 = bj * 2 + 1;

            // Fast K loop: full 32-element K blocks with direct A loads
            if (can_direct_load_a) {
                for (simsimd_size_t bk = 0; bk < full_k_blocks; bk++) {
                    simsimd_size_t const k_offset = bk * 32;

                    // Direct A tile loads from source (no intermediate buffer!)
                    _tile_loadd(0, a + row_block * a_stride_elements + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride_elements + k_offset, (int)a_stride);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + bk;
                    simsimd_bf16_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_bf16;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + bk;
                    simsimd_bf16_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_bf16;

                    // Load B tiles and compute
                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbf16ps(4, 0, 2); // C00 += A0 × B0
                    _tile_dpbf16ps(5, 0, 3); // C01 += A0 × B1
                    _tile_dpbf16ps(6, 1, 2); // C10 += A1 × B0
                    _tile_dpbf16ps(7, 1, 3); // C11 += A1 × B1
                }
            }
            else {
                // Fallback for small stride - copy to aligned buffer
                SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
                SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

                for (simsimd_size_t bk = 0; bk < full_k_blocks; bk++) {
                    simsimd_size_t const k_offset = bk * 32;

                    // Copy A tiles to aligned buffers
                    for (simsimd_size_t r = 0; r < 16; r++) {
                        simsimd_bf16_t const *src_upper = a + (row_block + r) * a_stride_elements + k_offset;
                        simsimd_bf16_t const *src_lower = a + (row_block + 16 + r) * a_stride_elements + k_offset;
                        for (simsimd_size_t c = 0; c < 32; c++) {
                            a_tile_upper[r][c] = src_upper[c];
                            a_tile_lower[r][c] = src_lower[c];
                        }
                    }

                    _tile_loadd(0, a_tile_upper, 64);
                    _tile_loadd(1, a_tile_lower, 64);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + bk;
                    simsimd_bf16_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_bf16;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + bk;
                    simsimd_bf16_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_bf16;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }

            // Handle K remainder (if k % 32 != 0)
            if (k_remainder > 0) {
                simsimd_size_t const k_offset = full_k_blocks * 32;
                SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
                SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

                // Use masked load for partial K block
                _simsimd_load_a_tile_bf16_masked(a + row_block * a_stride_elements + k_offset, a_stride_elements, 16,
                                                 k_remainder, (simsimd_bf16_t *)a_tile_upper);
                _simsimd_load_a_tile_bf16_masked(a + (row_block + 16) * a_stride_elements + k_offset, a_stride_elements,
                                                 16, k_remainder, (simsimd_bf16_t *)a_tile_lower);

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // B tiles for K remainder
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)full_k_blocks);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + full_k_blocks;
                simsimd_bf16_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_bf16;

                simsimd_size_t morton_idx1 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)full_k_blocks);
                if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + full_k_blocks;
                simsimd_bf16_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_bf16;

                _tile_loadd(2, b_tile_ptr0, 64);
                _tile_loadd(3, b_tile_ptr1, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
            }

            // Direct C stores (no intermediate buffer for full tiles!)
            if (can_direct_store_c) {
                _tile_stored(4, c + row_block * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(5, c + row_block * c_stride_elements + col_block + 16, (int)c_stride);
                _tile_stored(6, c + (row_block + 16) * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(7, c + (row_block + 16) * c_stride_elements + col_block + 16, (int)c_stride);
            }
            else {
                // Fallback for small stride - store via buffer
                SIMSIMD_ALIGN64 simsimd_f32_t c_tile[16][16];
                _tile_stored(4, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + r) * c_stride_elements + col_block + cc] = c_tile[r][cc];
                _tile_stored(5, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + r) * c_stride_elements + col_block + 16 + cc] = c_tile[r][cc];
                _tile_stored(6, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + 16 + r) * c_stride_elements + col_block + cc] = c_tile[r][cc];
                _tile_stored(7, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + 16 + r) * c_stride_elements + col_block + 16 + cc] = c_tile[r][cc];
            }
        }
    }

    // ========== SLOW PATH: Edge blocks (remainder rows and columns) ==========
    // These blocks require bounds checking and use masked operations

    // Handle remaining columns (rightmost partial 32-col blocks) for full row blocks
    if (n > full_n_blocks * 32) {
        simsimd_size_t const col_block = full_n_blocks * 32;
        simsimd_size_t const valid_cols = n - col_block;
        simsimd_size_t const b_tile_n0 = full_n_blocks * 2;
        simsimd_size_t const b_tile_n1 = b_tile_n0 + 1;
        int const has_second_col_tile = (valid_cols > 16);

        for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
            simsimd_size_t const row_block = bi * 32;

            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

            // K loop for edge column block
            for (simsimd_size_t k_offset = 0; k_offset < k; k_offset += tile_cols_bf16) {
                simsimd_size_t const k_valid = (k_offset + tile_cols_bf16 <= k) ? tile_cols_bf16 : k - k_offset;
                simsimd_size_t const b_tile_k = k_offset / tile_cols_bf16;

                // Load A tiles (full rows, possibly partial K)
                if (k_valid == 32 && can_direct_load_a) {
                    _tile_loadd(0, a + row_block * a_stride_elements + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride_elements + k_offset, (int)a_stride);
                }
                else {
                    _simsimd_load_a_tile_bf16_masked(a + row_block * a_stride_elements + k_offset, a_stride_elements,
                                                     16, k_valid, (simsimd_bf16_t *)a_tile_upper);
                    _simsimd_load_a_tile_bf16_masked(a + (row_block + 16) * a_stride_elements + k_offset,
                                                     a_stride_elements, 16, k_valid, (simsimd_bf16_t *)a_tile_lower);
                    _tile_loadd(0, a_tile_upper, 64);
                    _tile_loadd(1, a_tile_lower, 64);
                }

                // B tiles
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)b_tile_k);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + b_tile_k;
                _tile_loadd(2, b_tiles + morton_idx0 * tile_elements_bf16, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);

                if (has_second_col_tile && b_tile_n1 < tiles_along_n) {
                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)b_tile_k);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + b_tile_k;
                    _tile_loadd(3, b_tiles + morton_idx1 * tile_elements_bf16, 64);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }

            // Store with masked operations for partial columns
            SIMSIMD_ALIGN64 simsimd_f32_t c_tile[16][16];
            simsimd_size_t cols_first = (valid_cols > 16) ? 16 : valid_cols;
            simsimd_size_t cols_second = (valid_cols > 16) ? valid_cols - 16 : 0;

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile, c + row_block * c_stride_elements + col_block,
                                             c_stride_elements, 16, cols_first);
            _tile_stored(6, c_tile, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                             c + (row_block + 16) * c_stride_elements + col_block, c_stride_elements,
                                             16, cols_first);

            if (cols_second > 0) {
                _tile_stored(5, c_tile, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                                 c + row_block * c_stride_elements + col_block + 16, c_stride_elements,
                                                 16, cols_second);
                _tile_stored(7, c_tile, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                                 c + (row_block + 16) * c_stride_elements + col_block + 16,
                                                 c_stride_elements, 16, cols_second);
            }
        }
    }

    // Handle remaining rows (bottom partial 32-row blocks)
    if (m > full_m_blocks * 32) {
        simsimd_size_t const row_block = full_m_blocks * 32;
        simsimd_size_t const valid_rows = m - row_block;
        simsimd_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
        simsimd_size_t const rows_lower = (valid_rows > 16) ? valid_rows - 16 : 0;

        // Process all column blocks for the remaining rows
        for (simsimd_size_t col_block = 0; col_block < n; col_block += 32) {
            simsimd_size_t const valid_cols = (col_block + 32 <= n) ? 32 : n - col_block;
            simsimd_size_t const cols_first = (valid_cols > 16) ? 16 : valid_cols;
            simsimd_size_t const cols_second = (valid_cols > 16) ? valid_cols - 16 : 0;
            simsimd_size_t const b_tile_n0 = col_block / tile_rows;
            simsimd_size_t const b_tile_n1 = b_tile_n0 + 1;

            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_upper[16][32] = {{0}};
            SIMSIMD_ALIGN64 simsimd_bf16_t a_tile_lower[16][32] = {{0}};

            // K loop for edge row block
            for (simsimd_size_t k_offset = 0; k_offset < k; k_offset += tile_cols_bf16) {
                simsimd_size_t const k_valid = (k_offset + tile_cols_bf16 <= k) ? tile_cols_bf16 : k - k_offset;
                simsimd_size_t const b_tile_k = k_offset / tile_cols_bf16;

                // Load A tiles with masked operations (partial rows)
                _simsimd_load_a_tile_bf16_masked(a + row_block * a_stride_elements + k_offset, a_stride_elements,
                                                 rows_upper, k_valid, (simsimd_bf16_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_bf16_masked(a + (row_block + 16) * a_stride_elements + k_offset,
                                                     a_stride_elements, rows_lower, k_valid,
                                                     (simsimd_bf16_t *)a_tile_lower);
                }
                else {
                    // Zero the lower tile
                    for (simsimd_size_t r = 0; r < 16; r++)
                        _mm512_store_si512((__m512i *)(a_tile_lower[r]), _mm512_setzero_si512());
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // B tiles
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)b_tile_k);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + b_tile_k;
                _tile_loadd(2, b_tiles + morton_idx0 * tile_elements_bf16, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);

                if (cols_second > 0 && b_tile_n1 < tiles_along_n) {
                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)b_tile_k);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + b_tile_k;
                    _tile_loadd(3, b_tiles + morton_idx1 * tile_elements_bf16, 64);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }

            // Store with masked operations for partial rows
            SIMSIMD_ALIGN64 simsimd_f32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile, c + row_block * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, cols_first);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                                 c + (row_block + 16) * c_stride_elements + col_block,
                                                 c_stride_elements, rows_lower, cols_first);
            }

            if (cols_second > 0) {
                _tile_stored(5, c_tile, 64);
                _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                                 c + row_block * c_stride_elements + col_block + 16, c_stride_elements,
                                                 rows_upper, cols_second);

                if (rows_lower > 0) {
                    _tile_stored(7, c_tile, 64);
                    _simsimd_store_c_tile_f32_masked((simsimd_f32_t *)c_tile,
                                                     c + (row_block + 16) * c_stride_elements + col_block + 16,
                                                     c_stride_elements, rows_lower, cols_second);
                }
            }
        }
    }
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = n * sizeof(bf16).
 */
SIMSIMD_PUBLIC void simsimd_matmul_bf16_compact_sapphire( //
    void *c, simsimd_size_t m, simsimd_size_t n,          //
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

/*  I8 → I32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Uses TDPBSSD for signed INT8 dot product with INT32 accumulation.
 *  INT8 tiles are 16×64 (1KB each).
 *
 *  Optimized with fast-path/slow-path split:
 *    - Fast path: Full 32×32 blocks with direct tile load/store (no intermediate buffers)
 *    - Slow path: Edge blocks using AVX-512 masked operations
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_sapphire(                //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k,          //
    simsimd_size_t a_stride, simsimd_size_t c_stride) {

    // AMX tile dimensions for INT8
    simsimd_size_t const tile_rows = 16;
    simsimd_size_t const tile_cols_i8 = 64;
    simsimd_size_t const tile_elements_i8 = 1024;

    _simsimd_amx_tile_configure_sapphire();

    simsimd_size_t const tiles_along_n = (n + tile_rows - 1) / tile_rows;
    simsimd_size_t const tiles_along_k = (k + tile_cols_i8 - 1) / tile_cols_i8;
    simsimd_size_t const total_tiles = tiles_along_n * tiles_along_k;
    simsimd_size_t const c_stride_elements = c_stride / sizeof(simsimd_i32_t);
    simsimd_i8_t const *b_tiles = (simsimd_i8_t const *)b_packed;

    // Compute full block counts for fast-path
    simsimd_size_t const full_m_blocks = m / 32;
    simsimd_size_t const full_n_blocks = n / 32;
    simsimd_size_t const full_k_blocks = k / 64;
    simsimd_size_t const k_remainder = k - full_k_blocks * 64;

    // Check if strides allow direct tile operations (need 64 bytes for I8 A tile row and I32 C tile row)
    int const can_direct_load_a = (a_stride >= 64);
    int const can_direct_store_c = (c_stride >= 64);

    // ========== FAST PATH: Full 32×32 blocks ==========
    // No bounds checking needed in inner loops - all tiles are complete
    for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
        simsimd_size_t const row_block = bi * 32;

        for (simsimd_size_t bj = 0; bj < full_n_blocks; bj++) {
            simsimd_size_t const col_block = bj * 32;

            // Zero accumulator tiles
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Get B tile indices for this column block
            simsimd_size_t const b_tile_n0 = bj * 2;
            simsimd_size_t const b_tile_n1 = bj * 2 + 1;

            // Fast K loop: full 64-element K blocks with direct A loads
            if (can_direct_load_a) {
                for (simsimd_size_t bk = 0; bk < full_k_blocks; bk++) {
                    simsimd_size_t const k_offset = bk * 64;

                    // Direct A tile loads from source (no intermediate buffer!)
                    _tile_loadd(0, a + row_block * a_stride + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride + k_offset, (int)a_stride);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + bk;
                    simsimd_i8_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_i8;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + bk;
                    simsimd_i8_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_i8;

                    // Load B tiles and compute
                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbssd(4, 0, 2); // C00 += A0 × B0
                    _tile_dpbssd(5, 0, 3); // C01 += A0 × B1
                    _tile_dpbssd(6, 1, 2); // C10 += A1 × B0
                    _tile_dpbssd(7, 1, 3); // C11 += A1 × B1
                }
            }
            else {
                // Fallback for small stride - copy to aligned buffer
                SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
                SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

                for (simsimd_size_t bk = 0; bk < full_k_blocks; bk++) {
                    simsimd_size_t const k_offset = bk * 64;

                    // Copy A tiles to aligned buffers
                    for (simsimd_size_t r = 0; r < 16; r++) {
                        simsimd_i8_t const *src_upper = a + (row_block + r) * a_stride + k_offset;
                        simsimd_i8_t const *src_lower = a + (row_block + 16 + r) * a_stride + k_offset;
                        for (simsimd_size_t c = 0; c < 64; c++) {
                            a_tile_upper[r][c] = src_upper[c];
                            a_tile_lower[r][c] = src_lower[c];
                        }
                    }

                    _tile_loadd(0, a_tile_upper, 64);
                    _tile_loadd(1, a_tile_lower, 64);

                    // B tiles via Morton indexing
                    simsimd_size_t morton_idx0 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)bk);
                    if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + bk;
                    simsimd_i8_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_i8;

                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)bk);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + bk;
                    simsimd_i8_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_i8;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
            }

            // Handle K remainder (if k % 64 != 0)
            if (k_remainder > 0) {
                simsimd_size_t const k_offset = full_k_blocks * 64;
                SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
                SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

                // Use masked load for partial K block
                _simsimd_load_a_tile_i8_masked(a + row_block * a_stride + k_offset, a_stride, 16, k_remainder,
                                               (simsimd_i8_t *)a_tile_upper);
                _simsimd_load_a_tile_i8_masked(a + (row_block + 16) * a_stride + k_offset, a_stride, 16, k_remainder,
                                               (simsimd_i8_t *)a_tile_lower);

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // B tiles for K remainder
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)full_k_blocks);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + full_k_blocks;
                simsimd_i8_t const *b_tile_ptr0 = b_tiles + morton_idx0 * tile_elements_i8;

                simsimd_size_t morton_idx1 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)full_k_blocks);
                if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + full_k_blocks;
                simsimd_i8_t const *b_tile_ptr1 = b_tiles + morton_idx1 * tile_elements_i8;

                _tile_loadd(2, b_tile_ptr0, 64);
                _tile_loadd(3, b_tile_ptr1, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }

            // Direct C stores (no intermediate buffer for full tiles!)
            if (can_direct_store_c) {
                _tile_stored(4, c + row_block * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(5, c + row_block * c_stride_elements + col_block + 16, (int)c_stride);
                _tile_stored(6, c + (row_block + 16) * c_stride_elements + col_block, (int)c_stride);
                _tile_stored(7, c + (row_block + 16) * c_stride_elements + col_block + 16, (int)c_stride);
            }
            else {
                // Fallback for small stride - store via buffer
                SIMSIMD_ALIGN64 simsimd_i32_t c_tile[16][16];
                _tile_stored(4, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + r) * c_stride_elements + col_block + cc] = c_tile[r][cc];
                _tile_stored(5, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + r) * c_stride_elements + col_block + 16 + cc] = c_tile[r][cc];
                _tile_stored(6, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + 16 + r) * c_stride_elements + col_block + cc] = c_tile[r][cc];
                _tile_stored(7, c_tile, 64);
                for (simsimd_size_t r = 0; r < 16; r++)
                    for (simsimd_size_t cc = 0; cc < 16; cc++)
                        c[(row_block + 16 + r) * c_stride_elements + col_block + 16 + cc] = c_tile[r][cc];
            }
        }
    }

    // ========== SLOW PATH: Edge blocks (remainder rows and columns) ==========
    // These blocks require bounds checking and use masked operations

    // Handle remaining columns (rightmost partial 32-col blocks) for full row blocks
    if (n > full_n_blocks * 32) {
        simsimd_size_t const col_block = full_n_blocks * 32;
        simsimd_size_t const valid_cols = n - col_block;
        simsimd_size_t const b_tile_n0 = full_n_blocks * 2;
        simsimd_size_t const b_tile_n1 = b_tile_n0 + 1;
        int const has_second_col_tile = (valid_cols > 16);

        for (simsimd_size_t bi = 0; bi < full_m_blocks; bi++) {
            simsimd_size_t const row_block = bi * 32;

            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

            // K loop for edge column block
            for (simsimd_size_t k_offset = 0; k_offset < k; k_offset += tile_cols_i8) {
                simsimd_size_t const k_valid = (k_offset + tile_cols_i8 <= k) ? tile_cols_i8 : k - k_offset;
                simsimd_size_t const b_tile_k = k_offset / tile_cols_i8;

                // Load A tiles (full rows, possibly partial K)
                if (k_valid == 64 && can_direct_load_a) {
                    _tile_loadd(0, a + row_block * a_stride + k_offset, (int)a_stride);
                    _tile_loadd(1, a + (row_block + 16) * a_stride + k_offset, (int)a_stride);
                }
                else {
                    _simsimd_load_a_tile_i8_masked(a + row_block * a_stride + k_offset, a_stride, 16, k_valid,
                                                   (simsimd_i8_t *)a_tile_upper);
                    _simsimd_load_a_tile_i8_masked(a + (row_block + 16) * a_stride + k_offset, a_stride, 16, k_valid,
                                                   (simsimd_i8_t *)a_tile_lower);
                    _tile_loadd(0, a_tile_upper, 64);
                    _tile_loadd(1, a_tile_lower, 64);
                }

                // B tiles
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)b_tile_k);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + b_tile_k;
                _tile_loadd(2, b_tiles + morton_idx0 * tile_elements_i8, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);

                if (has_second_col_tile && b_tile_n1 < tiles_along_n) {
                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)b_tile_k);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + b_tile_k;
                    _tile_loadd(3, b_tiles + morton_idx1 * tile_elements_i8, 64);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(7, 1, 3);
                }
            }

            // Store with masked operations for partial columns
            SIMSIMD_ALIGN64 simsimd_i32_t c_tile[16][16];
            simsimd_size_t cols_first = (valid_cols > 16) ? 16 : valid_cols;
            simsimd_size_t cols_second = (valid_cols > 16) ? valid_cols - 16 : 0;

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile, c + row_block * c_stride_elements + col_block,
                                             c_stride_elements, 16, cols_first);
            _tile_stored(6, c_tile, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                             c + (row_block + 16) * c_stride_elements + col_block, c_stride_elements,
                                             16, cols_first);

            if (cols_second > 0) {
                _tile_stored(5, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + row_block * c_stride_elements + col_block + 16, c_stride_elements,
                                                 16, cols_second);
                _tile_stored(7, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + (row_block + 16) * c_stride_elements + col_block + 16,
                                                 c_stride_elements, 16, cols_second);
            }
        }
    }

    // Handle remaining rows (bottom partial 32-row blocks)
    if (m > full_m_blocks * 32) {
        simsimd_size_t const row_block = full_m_blocks * 32;
        simsimd_size_t const valid_rows = m - row_block;
        simsimd_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
        simsimd_size_t const rows_lower = (valid_rows > 16) ? valid_rows - 16 : 0;

        // Process all column blocks for the remaining rows
        for (simsimd_size_t col_block = 0; col_block < n; col_block += 32) {
            simsimd_size_t const valid_cols = (col_block + 32 <= n) ? 32 : n - col_block;
            simsimd_size_t const cols_first = (valid_cols > 16) ? 16 : valid_cols;
            simsimd_size_t const cols_second = (valid_cols > 16) ? valid_cols - 16 : 0;
            simsimd_size_t const b_tile_n0 = col_block / tile_rows;
            simsimd_size_t const b_tile_n1 = b_tile_n0 + 1;

            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_upper[16][64] = {{0}};
            SIMSIMD_ALIGN64 simsimd_i8_t a_tile_lower[16][64] = {{0}};

            // K loop for edge row block
            for (simsimd_size_t k_offset = 0; k_offset < k; k_offset += tile_cols_i8) {
                simsimd_size_t const k_valid = (k_offset + tile_cols_i8 <= k) ? tile_cols_i8 : k - k_offset;
                simsimd_size_t const b_tile_k = k_offset / tile_cols_i8;

                // Load A tiles with masked operations (partial rows)
                _simsimd_load_a_tile_i8_masked(a + row_block * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                               (simsimd_i8_t *)a_tile_upper);
                if (rows_lower > 0) {
                    _simsimd_load_a_tile_i8_masked(a + (row_block + 16) * a_stride + k_offset, a_stride, rows_lower,
                                                   k_valid, (simsimd_i8_t *)a_tile_lower);
                }
                else {
                    // Zero the lower tile
                    for (simsimd_size_t r = 0; r < 16; r++)
                        _mm512_store_si512((__m512i *)(a_tile_lower[r]), _mm512_setzero_si512());
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // B tiles
                simsimd_size_t morton_idx0 =
                    _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n0, (simsimd_u32_t)b_tile_k);
                if (morton_idx0 >= total_tiles) morton_idx0 = b_tile_n0 * tiles_along_k + b_tile_k;
                _tile_loadd(2, b_tiles + morton_idx0 * tile_elements_i8, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);

                if (cols_second > 0 && b_tile_n1 < tiles_along_n) {
                    simsimd_size_t morton_idx1 =
                        _simsimd_morton_encode_sapphire((simsimd_u32_t)b_tile_n1, (simsimd_u32_t)b_tile_k);
                    if (morton_idx1 >= total_tiles) morton_idx1 = b_tile_n1 * tiles_along_k + b_tile_k;
                    _tile_loadd(3, b_tiles + morton_idx1 * tile_elements_i8, 64);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(7, 1, 3);
                }
            }

            // Store with masked operations for partial rows
            SIMSIMD_ALIGN64 simsimd_i32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile, c + row_block * c_stride_elements + col_block,
                                             c_stride_elements, rows_upper, cols_first);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + (row_block + 16) * c_stride_elements + col_block,
                                                 c_stride_elements, rows_lower, cols_first);
            }

            if (cols_second > 0) {
                _tile_stored(5, c_tile, 64);
                _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                 c + row_block * c_stride_elements + col_block + 16, c_stride_elements,
                                                 rows_upper, cols_second);

                if (rows_lower > 0) {
                    _tile_stored(7, c_tile, 64);
                    _simsimd_store_c_tile_i32_masked((simsimd_i32_t *)c_tile,
                                                     c + (row_block + 16) * c_stride_elements + col_block + 16,
                                                     c_stride_elements, rows_lower, cols_second);
                }
            }
        }
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 * rsqrt(a_norm[i] * b_norm[j])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = n * sizeof(i8).
 */
SIMSIMD_PUBLIC void simsimd_matmul_i8_compact_sapphire( //
    void *c, simsimd_size_t m, simsimd_size_t n,        //
    simsimd_size_t c_stride,                            //
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
#endif // SIMSIMD_TARGET_SAPPHIRE

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
