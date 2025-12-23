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
SIMSIMD_PUBLIC void simsimd_matmul_bf16_serial(                       //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_bf16_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_serial(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_serial(             //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_serial(                  //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_serial(                     //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i8_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

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
SIMSIMD_PUBLIC void simsimd_matmul_i8_genoa(                      //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i8_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

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
SIMSIMD_PUBLIC void simsimd_matmul_bf16_genoa(                        //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_bf16_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

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
SIMSIMD_PUBLIC void simsimd_matmul_bf16_sapphire(                     //
    simsimd_bf16_t const *a, void const *b_packed, simsimd_bf16_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

SIMSIMD_PUBLIC simsimd_size_t simsimd_matmul_i8_packed_size_sapphire(simsimd_size_t n, simsimd_size_t k);
SIMSIMD_PUBLIC void simsimd_matmul_i8_pack_sapphire(           //
    simsimd_i8_t const *b, simsimd_size_t n, simsimd_size_t k, //
    simsimd_size_t b_stride, void *b_packed);
SIMSIMD_PUBLIC void simsimd_matmul_i8_i32_sapphire(                //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i32_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);
SIMSIMD_PUBLIC void simsimd_matmul_i8_sapphire(                   //
    simsimd_i8_t const *a, void const *b_packed, simsimd_i8_t *c, //
    simsimd_size_t m, simsimd_size_t n, simsimd_size_t k, simsimd_size_t a_stride, simsimd_size_t c_stride);

/*  Legacy macro-generated serial implementations for backward compatibility.
 *  These operate on unpacked matrices directly (no two-phase API).
 */
#define SIMSIMD_MAKE_MATMUL(name, input_type, accumulator_type, output_type, load_and_convert, convert_and_store) \
    void simsimd_matmul_##input_type##_##name(simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols,  \
                                              simsimd_##input_type##_t const *a, simsimd_size_t a_stride,         \
                                              simsimd_##input_type##_t const *b, simsimd_size_t b_stride,         \
                                              simsimd_##output_type##_t *c, simsimd_size_t c_stride) {            \
        for (simsimd_size_t i = 0; i < a_rows; ++i) {                                                             \
            simsimd_##input_type##_t const *a_row =                                                               \
                (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)a, i * a_stride);             \
            simsimd_##output_type##_t *c_row =                                                                    \
                (simsimd_##output_type##_t *)_simsimd_advance_by_bytes((void *)c, i * c_stride);                  \
            for (simsimd_size_t j = 0; j < b_rows; ++j) {                                                         \
                simsimd_##input_type##_t const *b_row =                                                           \
                    (simsimd_##input_type##_t const *)_simsimd_advance_by_bytes((void *)b, j * b_stride);         \
                simsimd_##accumulator_type##_t sum = 0;                                                           \
                for (simsimd_size_t k = 0; k < cols; ++k) {                                                       \
                    simsimd_##accumulator_type##_t aik = load_and_convert(a_row + k);                             \
                    simsimd_##accumulator_type##_t bjk = load_and_convert(b_row + k);                             \
                    sum += aik * bjk;                                                                             \
                }                                                                                                 \
                convert_and_store(sum, c_row + j);                                                                \
            }                                                                                                     \
        }                                                                                                         \
    }

#define SIMSIMD_MAKE_TILED(name, input_type, accumulator_type, output_type, load_and_convert, convert_and_store,      \
                           tile_size)                                                                                 \
    void simsimd_matmul_##input_type##_##name(simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols,      \
                                              simsimd_##input_type##_t const *a, simsimd_size_t a_stride,             \
                                              simsimd_##input_type##_t const *b, simsimd_size_t b_stride,             \
                                              simsimd_##output_type##_t *c, simsimd_size_t c_stride) {                \
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

SIMSIMD_MAKE_TILED(serial, f64, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f64_serial
SIMSIMD_MAKE_TILED(serial, f32, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f32_serial
SIMSIMD_MAKE_TILED(serial, f16, f32, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)     // simsimd_matmul_f16_serial
SIMSIMD_MAKE_TILED(serial, bf16, f32, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16, 16) // simsimd_matmul_bf16_serial
SIMSIMD_MAKE_TILED(serial, i8, i64, i8, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)          // simsimd_matmul_i8_serial
SIMSIMD_MAKE_TILED(accurate, f32, f64, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)      // simsimd_matmul_f32_accurate
SIMSIMD_MAKE_TILED(accurate, f16, f64, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)   // simsimd_matmul_f16_accurate
SIMSIMD_MAKE_TILED(accurate, bf16, f64, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16,
                   16) // simsimd_matmul_bf16_accurate

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

#if SIMSIMD_TARGET_X86
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
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), apply_to = function)

// Multiple levels of tiling are used here:
// - AMX tile size: (16 x 32) for BF16, 1 KB per each of 8 tile registers (TMM0-TMM7).
// - L1 cache tile: fits 3 matrices (A, B in BF16, C in F32) plus accumulators.
//   (128 x 128) BF16 tile = 32 KB, so A & B = 64 KB total.
//   (128 x 128) F32 tile = 64 KB for C accumulator.
//   This totals ~192 KB, fitting comfortably in L1 cache.

SIMSIMD_PUBLIC void simsimd_matmul_bf16_sapphire(                      //
    simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols, //
    simsimd_bf16_t const *a, simsimd_size_t a_row_stride_bytes,        //
    simsimd_bf16_t const *b, simsimd_size_t b_row_stride_bytes,        //
    simsimd_bf16_t *c, simsimd_size_t c_row_stride_bytes) {

    SIMSIMD_ALIGN64 simsimd_bf16_t a_l1_tile[SIMSIMD_MATMUL_L1_TILE_M][SIMSIMD_MATMUL_L1_TILE_M];
    SIMSIMD_ALIGN64 simsimd_bf16_t b_l1_tile[SIMSIMD_MATMUL_L1_TILE_M][SIMSIMD_MATMUL_L1_TILE_M];
    SIMSIMD_ALIGN64 simsimd_f32_t c_l1_tile[SIMSIMD_MATMUL_L1_TILE_M][SIMSIMD_MATMUL_L1_TILE_M];
    SIMSIMD_ALIGN64 simsimd_f32_t a2_l1_tile[SIMSIMD_MATMUL_L1_TILE_M][SIMSIMD_MATMUL_L1_TILE_M];
    SIMSIMD_ALIGN64 simsimd_f32_t b2_l1_tile[SIMSIMD_MATMUL_L1_TILE_M][SIMSIMD_MATMUL_L1_TILE_M];

    SIMSIMD_STATIC_ASSERT(SIMSIMD_MATMUL_L1_TILE_M % 32 == 0, SIMSIMD_MATMUL_L1_TILE_M_NOT_MULTIPLE_OF_32);

    // Set up the AMX tile configuration structure.
    // There are 8 tile registers from TMM0 to TMM7. Each is 16 rows by 64 bytes, fitting up to 1 KB of data.
    // The actual dimensions can be different and are controlled by `rows` and `colsb` - the width in bytes.
    SIMSIMD_ALIGN64 simsimd_u8_t amx_tiles_config[64];
    _mm512_store_si512((__m512i *)amx_tiles_config, _mm512_setzero_si512()); // Will fail, if the buffer is not aligned.
    simsimd_u8_t *amx_palette_id_ptr = &amx_tiles_config[0];
    simsimd_u16_t *amx_tiles_colsb_ptr = (simsimd_u16_t *)(&amx_tiles_config[16]); //! 16-bit integers!
    simsimd_u8_t *amx_tiles_rows_ptr = &amx_tiles_config[48];                      //! 8-bit integers!
    *amx_palette_id_ptr = 1; // The only palette currently supported

    // When using AMX tiles, we generally want to minimize the number of loads and stores,
    // especially for the second argument, as it will involve reordering.
    // So ideally, we want to load any AMX tile of B just once, and multiply it by many AMX tiles of A.
    // That way we are computing a vertical band of C. Let's assign:
    // - A tiles: TMM0, TMM1 - for (16 x 32) `bf16` values.
    // - B tiles: TMM2, TMM3 - for (16 x 32) `bf16` values.
    // - C tiles: TMM4, TMM5, TMM6, TMM7 - for (16 x 16) `f32` values.
    amx_tiles_rows_ptr[0] = amx_tiles_rows_ptr[1] = amx_tiles_rows_ptr[2] = amx_tiles_rows_ptr[3] =
        amx_tiles_rows_ptr[4] = amx_tiles_rows_ptr[5] = amx_tiles_rows_ptr[6] = amx_tiles_rows_ptr[7] = 16;
    amx_tiles_colsb_ptr[0] = amx_tiles_colsb_ptr[1] = amx_tiles_colsb_ptr[2] = amx_tiles_colsb_ptr[3] =
        amx_tiles_colsb_ptr[4] = amx_tiles_colsb_ptr[5] = amx_tiles_colsb_ptr[6] = amx_tiles_colsb_ptr[7] = 64;
    _tile_loadconfig(&amx_tiles_config);
    _tile_zero(4); // C top left tile.
    _tile_zero(5); // C top right tile.
    _tile_zero(6); // C bottom left tile.
    _tile_zero(7); // C bottom right tile.

    for (simsimd_size_t a_l1_start_row = 0; a_l1_start_row < a_rows; a_l1_start_row += SIMSIMD_MATMUL_L1_TILE_M) {
        /// Below comes code for rows: A[a_l1_start_row : a_l1_start_row + a_l1_count_rows].
        simsimd_size_t const a_l1_count_rows =
            (a_l1_start_row + SIMSIMD_MATMUL_L1_TILE_M < a_rows) ? SIMSIMD_MATMUL_L1_TILE_M : a_rows - a_l1_start_row;

        for (simsimd_size_t b_l1_start_row = 0; b_l1_start_row != b_rows; b_l1_start_row += SIMSIMD_MATMUL_L1_TILE_M) {
            /// Below comes code for rows: B[b_l1_start_row : b_l1_start_row + b_l1_count_rows]
            simsimd_size_t const b_l1_count_rows = (b_l1_start_row + SIMSIMD_MATMUL_L1_TILE_M < b_rows)
                                                       ? SIMSIMD_MATMUL_L1_TILE_M
                                                       : b_rows - b_l1_start_row;

            // Load the existing values in C tile:
            // C[a_l1_start_row:a_l1_start_row+a_l1_count_rows][b_l1_start_row:b_l1_start_row+b_l1_count_rows].
            // Piece of cake with AVX-512, knowing that the data is already aligned.
            // Each ZMM register can hold 64 bytes, so we can load 16x BF16 upcasting to 16x F32 elements at once.
            simsimd_size_t const c_tail_size = b_l1_count_rows % 16;
            // If the `b_l1_count_rows` is not divisible by 16, handle the tail with masked loads in AVX-512.
            __mmask16 const c_tail_mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, c_tail_size);
            for (simsimd_size_t row_in_l1 = 0; row_in_l1 < a_l1_count_rows; ++row_in_l1) {
                simsimd_size_t col_in_l1 = 0;
                simsimd_bf16_t const *c_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes(
                    (void *)(c + b_l1_start_row),                       // shift within a row
                    c_row_stride_bytes * (a_l1_start_row + row_in_l1)); // shift to the right row

                for (; col_in_l1 + 16 <= b_l1_count_rows; col_in_l1 += 16, c_global += 16)
                    _mm512_store_ps(&c_l1_tile[row_in_l1][col_in_l1],
                                    _simsimd_bf16x16_to_f32x16_skylake(_mm256_lddqu_si256((__m256i *)c_global)));
                if (c_tail_size)
                    _mm512_store_ps(
                        &c_l1_tile[row_in_l1][col_in_l1],
                        _simsimd_bf16x16_to_f32x16_skylake(_mm256_maskz_loadu_epi16(c_tail_mask, c_global)));
            }

            // At this point we are multiplying a horizontal band of A by a horizontal band of B.
            // Both will have up to `SIMSIMD_MATMUL_L1_TILE_M` rows and `cols` columns.
            for (simsimd_size_t l1_start_col = 0; l1_start_col < cols; l1_start_col += SIMSIMD_MATMUL_L1_TILE_M) {
                /// Below comes code for tiles:
                /// A[a_l1_start_row : a_l1_start_row + a_l1_count_rows, l1_start_col : l1_start_col + l1_count_cols]
                /// B[b_l1_start_row : b_l1_start_row + b_l1_count_rows, l1_start_col : l1_start_col + l1_count_cols]
                simsimd_size_t const l1_count_cols =
                    (l1_start_col + SIMSIMD_MATMUL_L1_TILE_M < cols) ? SIMSIMD_MATMUL_L1_TILE_M : cols - l1_start_col;

                // Now we need to load the tiles of A and B.
                // Piece of cake with AVX-512, knowing that the data is already aligned.
                // Each ZMM register can hold 64 bytes, so we can load 32x BF16 elements at once.
                int is_boundary_tile = (l1_start_col + SIMSIMD_MATMUL_L1_TILE_M > cols) ||
                                       (a_l1_start_row + SIMSIMD_MATMUL_L1_TILE_M > a_rows) ||
                                       (b_l1_start_row + SIMSIMD_MATMUL_L1_TILE_M > b_rows);
                if (!is_boundary_tile) {
                    // Load both A and B tiles in one loop.
                    for (simsimd_size_t row_in_l1 = 0; row_in_l1 != SIMSIMD_MATMUL_L1_TILE_M; ++row_in_l1) {
                        simsimd_bf16_t const *a_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes(
                            (void *)(a + l1_start_col),                         // shift within a row
                            a_row_stride_bytes * (a_l1_start_row + row_in_l1)); // shift to the right row

                        simsimd_bf16_t const *b_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes( //
                            (void *)(b + l1_start_col),                         // shift within a row
                            b_row_stride_bytes * (b_l1_start_row + row_in_l1)); // shift to the right row

                        for (simsimd_size_t col_in_l1 = 0; col_in_l1 != SIMSIMD_MATMUL_L1_TILE_M;
                             col_in_l1 += 32, a_global += 32, b_global += 32) {
                            _mm512_store_si512((__m512i *)&a_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_loadu_si512((__m512i *)a_global));
                            _mm512_store_si512((__m512i *)&b_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_loadu_si512((__m512i *)b_global));
                        }
                    }
                }
                // When dealing with boundary tiles, we need separate logic for the A and B tiles,
                // cause those matrices can have a different number of rows. We also need to take care
                // of the row tails, in case the number of columns is not divisible by the tile size.
                else {
                    simsimd_size_t const tail_size = l1_count_cols % 32;
                    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_size);
                    // Load the A tile.
                    for (simsimd_size_t row_in_l1 = 0; row_in_l1 != a_l1_count_rows; ++row_in_l1) {
                        simsimd_bf16_t const *a_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes(
                            (void *)(a + l1_start_col),                         // shift within a row
                            a_row_stride_bytes * (a_l1_start_row + row_in_l1)); // shift to the right row

                        simsimd_size_t col_in_l1 = 0;
                        for (; col_in_l1 + 32 < l1_count_cols; col_in_l1 += 32, a_global += 32)
                            _mm512_store_si512((__m512i *)&a_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_loadu_si512((__m512i *)a_global));
                        if (tail_size)
                            _mm512_store_si512(&a_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_maskz_loadu_epi16(tail_mask, a_global));
                    }
                    // Load the B tile.
                    for (simsimd_size_t row_in_l1 = 0; row_in_l1 != b_l1_count_rows; ++row_in_l1) {
                        simsimd_bf16_t const *b_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes(
                            (void *)(b + l1_start_col),                         // shift within a row
                            b_row_stride_bytes * (b_l1_start_row + row_in_l1)); // shift to the right row
                        simsimd_size_t col_in_l1 = 0;
                        for (; col_in_l1 + 32 < l1_count_cols; col_in_l1 += 32, b_global += 32)
                            _mm512_store_si512((__m512i *)&b_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_loadu_si512((__m512i *)b_global));
                        if (tail_size)
                            _mm512_store_si512(&b_l1_tile[row_in_l1][col_in_l1],
                                               _mm512_maskz_loadu_epi16(tail_mask, b_global));
                    }
                }

                // Now we need to view our `SIMSIMD_MATMUL_L1_TILE_M`-sided matrices
                // as composed of (16 x 16) tiles of 4-byte values, or (16 x 32) tiles of 2-byte values.
                // Vertically stacking TMM4 and TMM5, we can see that as a (32 x 32) "pivot" slice of B.
                simsimd_bf16_t tmm2_reordered[16][16][2];
                simsimd_bf16_t tmm3_reordered[16][16][2];
                for (simsimd_size_t b_row_in_l1 = 0; b_row_in_l1 != SIMSIMD_MATMUL_L1_TILE_M; b_row_in_l1 += 32) {
                    for (simsimd_size_t b_col_in_l1 = 0; b_col_in_l1 != SIMSIMD_MATMUL_L1_TILE_M; b_col_in_l1 += 32) {

                        // Load and permute the data from B for AMX, simultaneously transposing!
                        // TODO: Optimize with AVX-512.
                        for (simsimd_size_t col_in_amx_tile = 0; col_in_amx_tile != 32; ++col_in_amx_tile) {
                            for (simsimd_size_t row_in_amx_tile = 0; row_in_amx_tile != 16; ++row_in_amx_tile) {
                                tmm2_reordered[col_in_amx_tile / 2][row_in_amx_tile][col_in_amx_tile % 2] =
                                    b_l1_tile[b_row_in_l1 + row_in_amx_tile][b_col_in_l1 + col_in_amx_tile];
                                tmm3_reordered[col_in_amx_tile / 2][row_in_amx_tile][col_in_amx_tile % 2] =
                                    b_l1_tile[b_row_in_l1 + row_in_amx_tile + 16][b_col_in_l1 + col_in_amx_tile];
                            }
                        }
                        _tile_loadd(2, &tmm2_reordered[0][0][0], 64);
                        _tile_loadd(3, &tmm3_reordered[0][0][0], 64);

                        // Now we will walk through all the entries in the first 32 columns of the L1 tile of A.
                        // We will multiply them by TMM4 and TMM5, accumulating into TMM6 and TMM7 respectively.
                        for (simsimd_size_t a_row_in_l1 = 0; a_row_in_l1 != SIMSIMD_MATMUL_L1_TILE_M;
                             a_row_in_l1 += 32) {
                            _tile_loadd(0, &a_l1_tile[a_row_in_l1][b_col_in_l1],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_bf16_t));
                            _tile_loadd(1, &a_l1_tile[a_row_in_l1 + 16][b_col_in_l1],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_bf16_t));

                            _tile_loadd(4, &c_l1_tile[a_row_in_l1][b_row_in_l1],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_loadd(5, &c_l1_tile[a_row_in_l1][b_row_in_l1 + 16],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_loadd(6, &c_l1_tile[a_row_in_l1 + 16][b_row_in_l1],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_loadd(7, &c_l1_tile[a_row_in_l1 + 16][b_row_in_l1 + 16],
                                        SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));

                            // Perform all possible multiplications.
                            _tile_dpbf16ps(4, 0, 2);
                            _tile_dpbf16ps(5, 0, 3);
                            _tile_dpbf16ps(6, 1, 2);
                            _tile_dpbf16ps(7, 1, 3);

                            // Save back the updated C values.
                            _tile_stored(4, &c_l1_tile[a_row_in_l1][b_row_in_l1],
                                         SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_stored(5, &c_l1_tile[a_row_in_l1][b_row_in_l1 + 16],
                                         SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_stored(6, &c_l1_tile[a_row_in_l1 + 16][b_row_in_l1],
                                         SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                            _tile_stored(7, &c_l1_tile[a_row_in_l1 + 16][b_row_in_l1 + 16],
                                         SIMSIMD_MATMUL_L1_TILE_M * sizeof(simsimd_f32_t));
                        }
                    }
                }
            }

            // Export the C values to global memory:
            // C[a_l1_start_row:a_l1_start_row+a_l1_count_rows][b_l1_start_row:b_l1_start_row+b_l1_count_rows].
            for (simsimd_size_t row_in_l1 = 0; row_in_l1 < a_l1_count_rows; ++row_in_l1) {
                simsimd_size_t col_in_l1 = 0;
                simsimd_bf16_t const *c_global = (simsimd_bf16_t const *)_simsimd_advance_by_bytes(
                    (void *)(c + b_l1_start_row),                     // shift within a row
                    c_row_stride_bytes * (a_l1_start_row + row_in_l1) // shift to the right row
                );
                for (; col_in_l1 + 16 <= b_l1_count_rows; col_in_l1 += 16, c_global += 16)
                    _mm256_storeu_si256((__m256i *)c_global, _simsimd_f32x16_to_bf16x16_skylake(
                                                                 _mm512_load_ps(&c_l1_tile[row_in_l1][col_in_l1])));
                if (c_tail_size)
                    _mm256_mask_storeu_epi16(
                        (__m256i *)c_global, c_tail_mask,
                        _simsimd_f32x16_to_bf16x16_skylake(_mm512_load_ps(&c_l1_tile[row_in_l1][col_in_l1])));
            }
        }
    }
}

SIMSIMD_PUBLIC void simsimd_matmul_i8_sapphire(                        //
    simsimd_size_t a_rows, simsimd_size_t b_rows, simsimd_size_t cols, //
    simsimd_i8_t const *a, simsimd_size_t a_stride,                    //
    simsimd_i8_t const *b, simsimd_size_t b_stride,                    //
    simsimd_i8_t *c, simsimd_size_t c_stride) {}

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
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
