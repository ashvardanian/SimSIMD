/**
 *  @brief Portable GEMM macros and serial implementations for matrix multiplication.
 *  @file include/numkong/dots/serial.h
 *  @sa include/numkong/dots.h for API overview and use cases
 *  @author Ash Vardanian
 *
 *  This file provides two macro families for generating GEMM kernels:
 *
 *  - nk_define_dots_packed_: vectorized inner-products between rows of A and Bᵀ
 *  - nk_define_dots_symmetric_: vectorized inner-products between rows and columns of A
 *
 *  Both use the same B packing format (see below), enabling pack-once-use-anywhere.
 *
 *  @section packing B Matrix Packing Format
 *
 *  Computing C = A × Bᵀ where:
 *
 *  - A[row_count, depth] row-major: A[i, k] at address A + i × lda + k
 *  - B[column_count, depth] row-major (pre-transposed): B[j, k] at address B + j × ldb + k
 *  - C[row_count, column_count] row-major: C[i, j] at address C + i × ldc + j
 *
 *  The API convention stores B as Bᵀ for efficient SIMD access:
 *
 *  - A[i, k:k+4] is contiguous in row-major (good)
 *  - B[j, k:k+4] is contiguous in row-major (good - already transposed)
 *
 *  Packing adds row grouping (group_size = 16) for:
 *
 *  - Zero-padding on edges (avoids boundary checks in inner loop)
 *  - Cache-friendly blocking in outer loops
 *
 *  Memory layout example - B[8, 8] with 8 output columns (j), 8 depth (k):
 *
 *            k=0   k=1   k=2   k=3   k=4   k=5   k=6   k=7
 *         ┌─────────────────────────────────────────────────┐
 *    j=0  │  a0    a1    a2    a3    a4    a5    a6    a7   │
 *    j=1  │  b0    b1    b2    b3    b4    b5    b6    b7   │
 *    j=2  │  c0    c1    c2    c3    c4    c5    c6    c7   │
 *    j=3  │  d0    d1    d2    d3    d4    d5    d6    d7   │
 *    j=4  │  e0    e1    e2    e3    e4    e5    e6    e7   │
 *    j=5  │  f0    f1    f2    f3    f4    f5    f6    f7   │
 *    j=6  │  g0    g1    g2    g3    g4    g5    g6    g7   │
 *    j=7  │  h0    h1    h2    h3    h4    h5    h6    h7   │
 *         └─────────────────────────────────────────────────┘
 *
 *  Packed as B_packed[column_count_padded, depth] (grouped for alignment):
 *
 *    Group 0 (j=0..7, padded to 16):
 *      ┌───────────────────────────────────┐
 *      │ a0 a1 a2 a3 a4 a5 a6 a7 │  j=0    │  ← row 0 copied as-is
 *      │ b0 b1 b2 b3 b4 b5 b6 b7 │  j=1    │
 *      │ c0 c1 c2 c3 c4 c5 c6 c7 │  j=2    │
 *      │ d0 d1 d2 d3 d4 d5 d6 d7 │  j=3    │
 *      │ e0 e1 e2 e3 e4 e5 e6 e7 │  j=4    │
 *      │ f0 f1 f2 f3 f4 f5 f6 f7 │  j=5    │
 *      │ g0 g1 g2 g3 g4 g5 g6 g7 │  j=6    │
 *      │ h0 h1 h2 h3 h4 h5 h6 h7 │  j=7    │
 *      │ 00 00 00 00 00 00 00 00 │ padding │
 *      │ ...                     │ ...     │
 *      └───────────────────────────────────┘
 *
 *  Addressing formula for B_packed[j, k]:
 *
 *      group = j / group_size
 *      j_in_group = j % group_size
 *      B_packed[j, k] = packed[group * group_size * depth + j_in_group * depth + k]
 *
 *  Inner loop accesses B_packed[j, k:k+simd] which is contiguous - just ptr + k.
 */

#ifndef NK_DOTS_SERIAL_H
#define NK_DOTS_SERIAL_H
#include "numkong/types.h"
#include "numkong/set/serial.h"  // `nk_popcount_u1`
#include "numkong/cast/serial.h" // generic load/store helpers
#include "numkong/dot/serial.h"  // stateful dot product helpers

#if defined(__cplusplus)
extern "C" {
#endif

/*  Packed buffer header (64-byte aligned).
 *  Used by all packed matmul backends (serial, NEON, AVX-512, SVE).
 *
 *  Important units clarification:
 *  - For types where dimensions_per_value = 1 (f32, i8, u8, etc.): dimensions == values
 *  - For sub-byte types (i4x2, u4x2): dimensions ≠ values
 *    - dimensions = individual 4-bit nibbles (e.g., 128 nibbles)
 *    - values = storage bytes containing nibbles (e.g., 64 bytes for 128 nibbles)
 *    - dimensions_per_value = 2 (2 nibbles per byte)
 */
typedef struct {
    nk_u32_t column_count;        // Actual number of columns (not padded)
    nk_u32_t depth_dimensions;    // Logical depth in dimensions (nibbles for i4/u4, elements for i8/f32)
    nk_u32_t depth_padded_values; // Padded depth in storage values (bytes for i4/u4, elements for i8/f32)
    nk_u32_t reserved[13];        // Padding to 64 bytes
} nk_dots_packed_buffer_header_t;

/**
 *  @brief Generates function to calculate packed B matrix buffer size for GEMM micro-kernels.
 *
 *  Memory layout: B_packed[column_count, depth_padded] with header storing metadata.
 *  Buffer size: sizeof(header) + column_count × depth_padded × sizeof(storage_type)
 *  Depth padding logic: Round up to `depth_simd_dimensions` multiple, then add `depth_simd_dimensions`
 *  if stride is power-of-2.
 *
 *  @param suffix Platform suffix (serial, haswell, ice, etc.)
 *  @param input_type Original type of B matrix elements (may require conversion)
 *  @param storage_type Internal storage type in packed buffer (often f32 for mixed precision)
 *  @param output_type Result accumulator type (typically f32 or f64)
 *  @param depth_simd_dimensions SIMD vector width in elements for this platform/type combination
 */
#define nk_define_dots_pack_size_(name, suffix, input_type, storage_type, output_type, depth_simd_dimensions,    \
                                  dimensions_per_value)                                                          \
    NK_PUBLIC nk_size_t nk_dots_packed_size_##name##_##suffix(nk_size_t column_count, nk_size_t depth) {         \
        /* depth is always in logical dimensions (nibbles for i4, bytes for i8, etc.) */                         \
        /* depth_simd_dimensions is also in logical dimensions */                                                \
                                                                                                                 \
        /* Step 1: Pad depth in dimensions */                                                                    \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);         \
                                                                                                                 \
        /* Step 2: Convert dimensions to storage values */                                                       \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value); \
                                                                                                                 \
        /* Step 3: Calculate stride in bytes for power-of-2 check */                                             \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##storage_type##_t);                      \
                                                                                                                 \
        /* Step 4: Break power-of-2 strides for cache associativity */                                           \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                      \
            /* Add one SIMD step worth of storage values */                                                      \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);        \
        }                                                                                                        \
                                                                                                                 \
        /* Step 5: Return total buffer size in bytes */                                                          \
        return sizeof(nk_dots_packed_buffer_header_t) +                                                          \
               column_count * depth_values_padded * sizeof(nk_##storage_type##_t);                               \
    }

/**
 *  @brief Generates function to pack and optionally convert B matrix for efficient GEMM inner loops.
 *
 *  Packing serves two performance-critical purposes:
 *
 *  1. Type conversion (input_type → storage_type): For mixed-precision GEMM, convert B elements
 *     once during packing rather than repeatedly in tight inner loops. Example: F16 → F32 conversion
 *     happens once per element instead of once per (row of A × element of B) access. This amortizes
 *     conversion cost across all rows of A.
 *
 *  2. Cache optimization: Pad depth to break power-of-2 byte strides that cause cache associativity
 *     conflicts. Example: depth = 8192, F32 → stride = 32,768 bytes (power-of-2) maps to same cache sets,
 *     causing conflict misses. Padding to 8200 → stride = 32,800 bytes (non-power-of-2) distributes
 *     accesses across more cache sets.
 *
 *  Input layout: B[column_count, depth] stored row-major with b_stride_in_bytes between rows
 *  Output layout: B_packed[column_count, depth_padded] - simple column-major, no grouping
 *  Addressing: B_packed[j, k] = packed_data[j × depth_padded + k]
 *
 *  Depth padding: Round up to `depth_simd_dimensions` multiple, then add `depth_simd_dimensions`
 *  if stride is power-of-2. Zero-initializes entire buffer before copying to handle padding safely.
 *
 *  @param suffix Platform suffix (serial, haswell, ice, etc.)
 *  @param input_type Original type of B matrix elements (f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param storage_type Packed buffer element type (often f32 for mixed precision, same as input otherwise)
 *  @param output_type Result accumulator type (f32 or f64)
 *  @param scalar_convert_fn Element conversion function: void fn(input_type const*, storage_type*)
 *  @param depth_simd_dimensions SIMD vector width in elements for depth padding alignment
 */
#define nk_define_dots_pack_(name, suffix, input_type, storage_type, output_type, scalar_convert_fn,                 \
                             depth_simd_dimensions, dimensions_per_value)                                            \
    NK_PUBLIC void nk_dots_pack_##name##_##suffix(nk_##input_type##_t const *b, nk_size_t column_count,              \
                                                  nk_size_t depth, nk_size_t b_stride_in_bytes, void *b_packed) {    \
        /* Use identical padding calculation as pack_size */                                                         \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);             \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value);     \
                                                                                                                     \
        /* Power-of-2 breaking (same as pack_size) */                                                                \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##storage_type##_t);                          \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                          \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);            \
        }                                                                                                            \
                                                                                                                     \
        /* Calculate input depth in values */                                                                        \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                     \
                                                                                                                     \
        /* Store dimensions in header */                                                                             \
        nk_dots_packed_buffer_header_t *header = (nk_dots_packed_buffer_header_t *)b_packed;                         \
        header->column_count = (nk_u32_t)column_count;                                                               \
        header->depth_dimensions = (nk_u32_t)depth;                  /* depth in dimensions (nibbles for i4/u4) */   \
        header->depth_padded_values = (nk_u32_t)depth_values_padded; /* padded depth in VALUES (bytes for i4/u4) */  \
                                                                                                                     \
        nk_##storage_type##_t *packed = (nk_##storage_type##_t *)((char *)b_packed +                                 \
                                                                  sizeof(nk_dots_packed_buffer_header_t));           \
                                                                                                                     \
        /* Zero entire buffer for depth padding */                                                                   \
        nk_size_t const total_elements = column_count * depth_values_padded;                                         \
        for (nk_size_t i = 0; i < total_elements; ++i) packed[i] = 0;                                                \
                                                                                                                     \
        /* Copy/convert B[column_count, depth] to packed[column_count, depth_padded] - simple column-major */        \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                              \
            nk_##storage_type##_t *destination_row = packed + column_index * depth_values_padded;                    \
            nk_##input_type##_t const *source_row = (nk_##input_type##_t const *)((char const *)b +                  \
                                                                                  column_index * b_stride_in_bytes); \
            for (nk_size_t depth_index = 0; depth_index < depth_in_values; ++depth_index) {                          \
                scalar_convert_fn(&source_row[depth_index], &destination_row[depth_index]);                          \
            }                                                                                                        \
            /* Padding elements already zeroed above */                                                              \
        }                                                                                                            \
    }

/**
 *  @brief SIMD GEMM macro with simplified transpose-based B packing.
 *
 *  Computes C[row_count, column_count] = A[row_count, depth] × B[depth, column_count]
 *  where B is pre-packed as Bᵀ[column_count, depth].
 *
 *  Loop structure (GotoBLAS 5-loop design):
 *    Loop 1: column_block_size columns at a time (L3 blocking)
 *      Loop 2: depth_block depth at a time (L1 blocking)
 *        Loop 3: row_block_size rows at a time (L2 blocking)
 *          Loop 4: register_column_count columns (register tile)
 *            Loop 5: register_row_count rows (register tile)
 *              Loop 6: depth elements (SIMD accumulation) - tight inner loop
 *
 *  Inner loop structure (after B pointer setup outside depth-loop):
 *
 *      b_depth_ptr_0 = group_base_ptr + (column_index_in_group + 0) × depth;
 *      b_depth_ptr_1 = group_base_ptr + (column_index_in_group + 1) × depth;
 *      b_depth_ptr_2 = group_base_ptr + (column_index_in_group + 2) × depth;
 *      b_depth_ptr_3 = group_base_ptr + (column_index_in_group + 3) × depth;
 *
 *      for (depth_index = 0; depth_index < aligned_depth; depth_index += depth_simd_dimensions) {
 *          load_fn(b_depth_ptr_0 + depth_index, &b_vector_0);
 *          load_fn(b_depth_ptr_1 + depth_index, &b_vector_1);
 *          // ... 16 FMAs for 4 × 4 register tile
 *      }
 */
#define nk_define_dots_packed_4x4_vectors_aligned_(suffix, input_type, storage_type, output_type, vec_type,            \
                                                   state_type, result_vec_type, init_fn, load_a_fn, partial_load_a_fn, \
                                                   load_b_fn, partial_load_b_fn, update_fn, finalize_fn,               \
                                                   partial_store_fn, depth_simd_dimensions, dimensions_per_value)      \
                                                                                                                       \
    NK_PUBLIC void nk_dots_##suffix##_aligned_(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,       \
                                               nk_##output_type##_t *c_matrix, nk_size_t row_count,                    \
                                               nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,   \
                                               nk_size_t c_stride_in_bytes) {                                          \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_dots_packed_buffer_header_t const *header = (nk_dots_packed_buffer_header_t const *)b_packed_buffer;        \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        nk_##storage_type##_t const *packed_data =                                                                     \
            (nk_##storage_type##_t const *)((char const *)b_packed_buffer + sizeof(nk_dots_packed_buffer_header_t));   \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##storage_type##_t const *b_depth_ptr_0 = packed_data +                                         \
                                                                 (tile_column_start_index + 0) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_1 = packed_data +                                         \
                                                                 (tile_column_start_index + 1) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_2 = packed_data +                                         \
                                                                 (tile_column_start_index + 2) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_3 = packed_data +                                         \
                                                                 (tile_column_start_index + 3) * depth_padded;         \
                                                                                                                       \
                    /* Loop 4: Register tiling over rows (register_row_count rows per tile) */                         \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                                                                                                                       \
                        /* Initialize register_row_count × register_column_count accumulator states */                 \
                        state_type accumulator_tiles[4][4];                                                            \
                        init_fn(&accumulator_tiles[0][0]), init_fn(&accumulator_tiles[0][1]),                          \
                            init_fn(&accumulator_tiles[0][2]), init_fn(&accumulator_tiles[0][3]);                      \
                        init_fn(&accumulator_tiles[1][0]), init_fn(&accumulator_tiles[1][1]),                          \
                            init_fn(&accumulator_tiles[1][2]), init_fn(&accumulator_tiles[1][3]);                      \
                        init_fn(&accumulator_tiles[2][0]), init_fn(&accumulator_tiles[2][1]),                          \
                            init_fn(&accumulator_tiles[2][2]), init_fn(&accumulator_tiles[2][3]);                      \
                        init_fn(&accumulator_tiles[3][0]), init_fn(&accumulator_tiles[3][1]),                          \
                            init_fn(&accumulator_tiles[3][2]), init_fn(&accumulator_tiles[3][3]);                      \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_type##_t const *a_row_ptr_0 =                                                       \
                            (nk_##input_type##_t const *)((char const *)a_matrix +                                     \
                                                          (tile_row_start_index + 0) * a_stride_in_bytes);             \
                        nk_##input_type##_t const *a_row_ptr_1 =                                                       \
                            (nk_##input_type##_t const *)((char const *)a_matrix +                                     \
                                                          (tile_row_start_index + 1) * a_stride_in_bytes);             \
                        nk_##input_type##_t const *a_row_ptr_2 =                                                       \
                            (nk_##input_type##_t const *)((char const *)a_matrix +                                     \
                                                          (tile_row_start_index + 2) * a_stride_in_bytes);             \
                        nk_##input_type##_t const *a_row_ptr_3 =                                                       \
                            (nk_##input_type##_t const *)((char const *)a_matrix +                                     \
                                                          (tile_row_start_index + 3) * a_stride_in_bytes);             \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                                       \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                                       \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load next few elements from 4 rows from A (unpacked, may upcast) */                     \
                            load_a_fn(a_row_ptr_0 + depth_index, &a_vector_0);                                         \
                            load_a_fn(a_row_ptr_1 + depth_index, &a_vector_1);                                         \
                            load_a_fn(a_row_ptr_2 + depth_index, &a_vector_2);                                         \
                            load_a_fn(a_row_ptr_3 + depth_index, &a_vector_3);                                         \
                                                                                                                       \
                            /* Load next few elements from 4 rows from B (packed, already upcasted) */                 \
                            load_b_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                       \
                            load_b_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                       \
                            load_b_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                       \
                            load_b_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                       \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            update_fn(&accumulator_tiles[0][0], a_vector_0, b_vector_0,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[0][1], a_vector_0, b_vector_1,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[0][2], a_vector_0, b_vector_2,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[0][3], a_vector_0, b_vector_3,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[1][0], a_vector_1, b_vector_0,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[1][1], a_vector_1, b_vector_1,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[1][2], a_vector_1, b_vector_2,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[1][3], a_vector_1, b_vector_3,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[2][0], a_vector_2, b_vector_0,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[2][1], a_vector_2, b_vector_1,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[2][2], a_vector_2, b_vector_2,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[2][3], a_vector_2, b_vector_3,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[3][0], a_vector_3, b_vector_0,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[3][1], a_vector_3, b_vector_1,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[3][2], a_vector_3, b_vector_2,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                            update_fn(&accumulator_tiles[3][3], a_vector_3, b_vector_3,                                \
                                      depth_index * dimensions_per_value, depth_simd_dimensions);                      \
                        }                                                                                              \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        result_vec_type result_vector;                                                                 \
                        nk_##output_type##_t *c_row_ptr_0 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 0) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[0][0], &accumulator_tiles[0][1], &accumulator_tiles[0][2],      \
                                    &accumulator_tiles[0][3], &result_vector, depth);                                  \
                        partial_store_fn(&result_vector, c_row_ptr_0 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_1 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 1) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[1][0], &accumulator_tiles[1][1], &accumulator_tiles[1][2],      \
                                    &accumulator_tiles[1][3], &result_vector, depth);                                  \
                        partial_store_fn(&result_vector, c_row_ptr_1 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_2 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 2) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[2][0], &accumulator_tiles[2][1], &accumulator_tiles[2][2],      \
                                    &accumulator_tiles[2][3], &result_vector, depth);                                  \
                        partial_store_fn(&result_vector, c_row_ptr_2 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_3 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 3) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[3][0], &accumulator_tiles[3][1], &accumulator_tiles[3][2],      \
                                    &accumulator_tiles[3][3], &result_vector, depth);                                  \
                        partial_store_fn(&result_vector, c_row_ptr_3 + tile_column_start_index, 4);                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief 1 × 8 SIMD GEMM aligned kernel - processes 1 row × 8 columns per tile.
 *
 *  Optimized for cases where M is small relative to N (few rows, many columns).
 *  Uses same finalize function twice (for columns 0-3 and 4-7).
 */
#define nk_define_dots_packed_1x8_vectors_aligned_(suffix, input_type, storage_type, output_type, vec_type,            \
                                                   state_type, result_vec_type, init_fn, load_a_fn, partial_load_a_fn, \
                                                   load_b_fn, partial_load_b_fn, update_fn, finalize_fn,               \
                                                   partial_store_fn, depth_simd_dimensions, dimensions_per_value)      \
                                                                                                                       \
    NK_PUBLIC void nk_dots_##suffix##_1x8_aligned_(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,   \
                                                   nk_##output_type##_t *c_matrix, nk_size_t row_count,                \
                                                   nk_size_t column_count, nk_size_t depth,                            \
                                                   nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {         \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_dots_packed_buffer_header_t const *header = (nk_dots_packed_buffer_header_t const *)b_packed_buffer;        \
        nk_size_t const depth_padded = header->depth_padded_values; /* in storage values */                            \
                                                                                                                       \
        nk_##storage_type##_t const *packed_data =                                                                     \
            (nk_##storage_type##_t const *)((char const *)b_packed_buffer + sizeof(nk_dots_packed_buffer_header_t));   \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 1;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 8; /* Columns per register tile (2 × 4) */                             \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        (void)register_row_count; /* Used in comments, loop uses 1 directly */                                         \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t const row_block_end_index = row_block_start_index + row_block_size < row_count               \
                                                          ? row_block_start_index + row_block_size                     \
                                                          : row_count;                                                 \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##storage_type##_t const *b_depth_ptr_0 = packed_data +                                         \
                                                                 (tile_column_start_index + 0) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_1 = packed_data +                                         \
                                                                 (tile_column_start_index + 1) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_2 = packed_data +                                         \
                                                                 (tile_column_start_index + 2) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_3 = packed_data +                                         \
                                                                 (tile_column_start_index + 3) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_4 = packed_data +                                         \
                                                                 (tile_column_start_index + 4) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_5 = packed_data +                                         \
                                                                 (tile_column_start_index + 5) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_6 = packed_data +                                         \
                                                                 (tile_column_start_index + 6) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_7 = packed_data +                                         \
                                                                 (tile_column_start_index + 7) * depth_padded;         \
                                                                                                                       \
                    /* Loop 4: Process 1 row at a time */                                                              \
                    for (nk_size_t row_index = row_block_start_index; row_index < row_block_end_index; ++row_index) {  \
                                                                                                                       \
                        /* Initialize 1 × 8 accumulator states */                                                      \
                        state_type accumulator_0, accumulator_1, accumulator_2, accumulator_3, accumulator_4,          \
                            accumulator_5, accumulator_6, accumulator_7;                                               \
                        init_fn(&accumulator_0), init_fn(&accumulator_1), init_fn(&accumulator_2),                     \
                            init_fn(&accumulator_3), init_fn(&accumulator_4), init_fn(&accumulator_5),                 \
                            init_fn(&accumulator_6), init_fn(&accumulator_7);                                          \
                                                                                                                       \
                        /* A row pointer */                                                                            \
                        nk_##input_type##_t const *a_row_ptr =                                                         \
                            (nk_##input_type##_t const *)((char const *)a_matrix + row_index * a_stride_in_bytes);     \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector;                                                                             \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3, b_vector_4, b_vector_5, b_vector_6,   \
                            b_vector_7;                                                                                \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load A vector (1 row) */                                                                \
                            load_a_fn(a_row_ptr + depth_index, &a_vector);                                             \
                                                                                                                       \
                            /* Load B vectors (8 columns) */                                                           \
                            load_b_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                       \
                            load_b_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                       \
                            load_b_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                       \
                            load_b_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                       \
                            load_b_fn(b_depth_ptr_4 + depth_index, &b_vector_4);                                       \
                            load_b_fn(b_depth_ptr_5 + depth_index, &b_vector_5);                                       \
                            load_b_fn(b_depth_ptr_6 + depth_index, &b_vector_6);                                       \
                            load_b_fn(b_depth_ptr_7 + depth_index, &b_vector_7);                                       \
                                                                                                                       \
                            /* 8 FMAs: 1 A row × 8 B columns */                                                        \
                            update_fn(&accumulator_0, a_vector, b_vector_0, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_1, a_vector, b_vector_1, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_2, a_vector, b_vector_2, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_3, a_vector, b_vector_3, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_4, a_vector, b_vector_4, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_5, a_vector, b_vector_5, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_6, a_vector, b_vector_6, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_7, a_vector, b_vector_7, depth_index * dimensions_per_value,        \
                                      depth_simd_dimensions);                                                          \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store 1 × 8 results using two 4-way reductions */                              \
                        result_vec_type result_vector;                                                                 \
                        nk_##output_type##_t *c_row_ptr = (nk_##output_type##_t *)((char *)c_matrix +                  \
                                                                                   row_index * c_stride_in_bytes);     \
                        /* First 4 columns */                                                                          \
                        finalize_fn(&accumulator_0, &accumulator_1, &accumulator_2, &accumulator_3, &result_vector,    \
                                    depth);                                                                            \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index, 4);                      \
                        /* Second 4 columns */                                                                         \
                        finalize_fn(&accumulator_4, &accumulator_5, &accumulator_6, &accumulator_7, &result_vector,    \
                                    depth);                                                                            \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index + 4, 4);                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/* Generate both aligned kernel variants for each platform */
#define nk_define_dots_packed_(name, suffix, input_type, storage_type, output_type, vec_type, state_type,              \
                               result_vec_type, init_fn, load_a_fn, partial_load_a_fn, load_b_fn, partial_load_b_fn,   \
                               update_fn, finalize_fn, partial_store_fn, depth_simd_dimensions, dimensions_per_value)  \
    /* Generate 4 × 4 aligned kernel */                                                                                \
    nk_define_dots_packed_4x4_vectors_aligned_(name##_##suffix, input_type, storage_type, output_type, vec_type,       \
                                               state_type, result_vec_type, init_fn, load_a_fn, partial_load_a_fn,     \
                                               load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, \
                                               depth_simd_dimensions, dimensions_per_value)                            \
    /* Generate 1 × 8 aligned kernel */                                                                                \
    nk_define_dots_packed_1x8_vectors_aligned_(name##_##suffix, input_type, storage_type, output_type, vec_type,       \
                                               state_type, result_vec_type, init_fn, load_a_fn, partial_load_a_fn,     \
                                               load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, \
                                               depth_simd_dimensions, dimensions_per_value)                            \
                                                                                                                       \
    NK_PUBLIC void nk_dots_packed_##name##_##suffix(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,  \
                                                    nk_##output_type##_t *c_matrix, nk_size_t row_count,               \
                                                    nk_size_t column_count, nk_size_t depth,                           \
                                                    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {        \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_dots_packed_buffer_header_t const *header = (nk_dots_packed_buffer_header_t const *)b_packed_buffer;        \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        /* Cache blocking parameters (hardcoded for optimal L1/L2/L3 utilization) */                                   \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        (void)register_column_count;               /* Suppress unused warnings */                                      \
        /* Use 1 × 8 kernel when columns are aligned to 8 and many columns relative to rows */                         \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % depth_simd_dimensions == 0) {            \
            nk_dots_##name##_##suffix##_1x8_aligned_(a_matrix, b_packed_buffer, c_matrix, row_count, column_count,     \
                                                     depth, a_stride_in_bytes, c_stride_in_bytes);                     \
            return;                                                                                                    \
        }                                                                                                              \
        /* Use 4 × 4 kernel when dimensions are 4-aligned */                                                           \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_dots_##name##_##suffix##_aligned_(a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth,  \
                                                 a_stride_in_bytes, c_stride_in_bytes);                                \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Compute aligned/remainder depth for partial loads (correct for sub-byte types) */                           \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = (depth_dimensions_aligned + dimensions_per_value - 1) / dimensions_per_value;  \
        nk_size_t const depth_in_values = (depth + dimensions_per_value - 1) / dimensions_per_value;                   \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = (depth_simd_dimensions + dimensions_per_value - 1) / dimensions_per_value; \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        nk_##storage_type##_t const *packed_data =                                                                     \
            (nk_##storage_type##_t const *)((char const *)b_packed_buffer + sizeof(nk_dots_packed_buffer_header_t));   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 4: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                    nk_size_t tile_column_count = register_column_count;                                               \
                    if (tile_column_start_index + tile_column_count > column_block_end_index)                          \
                        tile_column_count = column_block_end_index - tile_column_start_index;                          \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##storage_type##_t const *b_depth_ptr_0 = packed_data +                                         \
                                                                 (tile_column_start_index + 0) * depth_padded;         \
                    nk_##storage_type##_t const *b_depth_ptr_1 = (tile_column_count > 1)                               \
                                                                     ? packed_data + (tile_column_start_index + 1) *   \
                                                                                         depth_padded                  \
                                                                     : b_depth_ptr_0;                                  \
                    nk_##storage_type##_t const *b_depth_ptr_2 = (tile_column_count > 2)                               \
                                                                     ? packed_data + (tile_column_start_index + 2) *   \
                                                                                         depth_padded                  \
                                                                     : b_depth_ptr_0;                                  \
                    nk_##storage_type##_t const *b_depth_ptr_3 = (tile_column_count > 3)                               \
                                                                     ? packed_data + (tile_column_start_index + 3) *   \
                                                                                         depth_padded                  \
                                                                     : b_depth_ptr_0;                                  \
                                                                                                                       \
                    /* Loop 5: Register tiling over rows (register_rows rows per tile) */                              \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                        nk_size_t tile_row_count = register_row_count;                                                 \
                        if (tile_row_start_index + tile_row_count > row_block_end_index)                               \
                            tile_row_count = row_block_end_index - tile_row_start_index;                               \
                                                                                                                       \
                        /* Initialize register_rows x register_cols accumulator states */                              \
                        state_type accumulator_tiles[4][4];                                                            \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            init_fn(&accumulator_tiles[r][0]);                                                         \
                            init_fn(&accumulator_tiles[r][1]);                                                         \
                            init_fn(&accumulator_tiles[r][2]);                                                         \
                            init_fn(&accumulator_tiles[r][3]);                                                         \
                        }                                                                                              \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_type##_t const *a_row_ptr_0 =                                                       \
                            (nk_##input_type##_t const *)((char const *)a_matrix +                                     \
                                                          (tile_row_start_index + 0) * a_stride_in_bytes);             \
                        nk_##input_type##_t const *a_row_ptr_1 =                                                       \
                            (tile_row_count > 1)                                                                       \
                                ? (nk_##input_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 1) * a_stride_in_bytes)        \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_type##_t const *a_row_ptr_2 =                                                       \
                            (tile_row_count > 2)                                                                       \
                                ? (nk_##input_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 2) * a_stride_in_bytes)        \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_type##_t const *a_row_ptr_3 =                                                       \
                            (tile_row_count > 3)                                                                       \
                                ? (nk_##input_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 3) * a_stride_in_bytes)        \
                                : a_row_ptr_0;                                                                         \
                                                                                                                       \
                        /* Tight inner loop: k elements with simple ptr+k addressing */                                \
                        vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                                 \
                        vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                                 \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            /* Load next few elements from 4 rows from A */                                            \
                            load_a_fn(a_row_ptr_0 + k, &a_first_vec);                                                  \
                            load_a_fn(a_row_ptr_1 + k, &a_second_vec);                                                 \
                            load_a_fn(a_row_ptr_2 + k, &a_third_vec);                                                  \
                            load_a_fn(a_row_ptr_3 + k, &a_fourth_vec);                                                 \
                                                                                                                       \
                            /* Load next few elements from 4 rows from B */                                            \
                            load_b_fn(b_depth_ptr_0 + k, &b_first_vec);                                                \
                            load_b_fn(b_depth_ptr_1 + k, &b_second_vec);                                               \
                            load_b_fn(b_depth_ptr_2 + k, &b_third_vec);                                                \
                            load_b_fn(b_depth_ptr_3 + k, &b_fourth_vec);                                               \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            update_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec, k * dimensions_per_value,    \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec, k * dimensions_per_value,    \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec, k * dimensions_per_value,  \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec, k * dimensions_per_value,  \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec, k * dimensions_per_value,    \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec, k * dimensions_per_value,    \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec, k * dimensions_per_value,  \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec, k * dimensions_per_value,   \
                                      depth_simd_dimensions);                                                          \
                            update_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec, k * dimensions_per_value,  \
                                      depth_simd_dimensions);                                                          \
                        }                                                                                              \
                                                                                                                       \
                        /* Handle remainder k positions with partial loads */                                          \
                        if (remainder_depth > 0) {                                                                     \
                            /* Load next few elements from 4 rows from A */                                            \
                            partial_load_a_fn(a_row_ptr_0 + aligned_depth, &a_first_vec, remainder_dimensions);        \
                            partial_load_a_fn(a_row_ptr_1 + aligned_depth, &a_second_vec, remainder_dimensions);       \
                            partial_load_a_fn(a_row_ptr_2 + aligned_depth, &a_third_vec, remainder_dimensions);        \
                            partial_load_a_fn(a_row_ptr_3 + aligned_depth, &a_fourth_vec, remainder_dimensions);       \
                                                                                                                       \
                            /* Load next few elements from 4 rows from B */                                            \
                            partial_load_b_fn(b_depth_ptr_0 + aligned_depth, &b_first_vec, remainder_dimensions);      \
                            partial_load_b_fn(b_depth_ptr_1 + aligned_depth, &b_second_vec, remainder_dimensions);     \
                            partial_load_b_fn(b_depth_ptr_2 + aligned_depth, &b_third_vec, remainder_dimensions);      \
                            partial_load_b_fn(b_depth_ptr_3 + aligned_depth, &b_fourth_vec, remainder_dimensions);     \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            update_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                              \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                              \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                            \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                            \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                              \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                              \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                            \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                             \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                            update_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                            \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            result_vec_type result_vector;                                                             \
                            finalize_fn(&accumulator_tiles[r][0], &accumulator_tiles[r][1], &accumulator_tiles[r][2],  \
                                        &accumulator_tiles[r][3], &result_vector, depth);                              \
                                                                                                                       \
                            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix +                  \
                                                                                   (tile_row_start_index + r) *        \
                                                                                       c_stride_in_bytes);             \
                            partial_store_fn(&result_vector, c_row + tile_column_start_index, tile_column_count);      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 * @brief Vectorized symmetric Gram matrix: C = A × Aᵀ
 *
 * Computes symmetric matrix C[i,j] = dot(A[i,:], A[j,:]) using vectorized operations.
 * Only computes upper triangle for efficiency, then mirrors to lower triangle.
 *
 * Optimizations:
 * - 16×16 register tiling (256 parallel accumulators)
 * - Vector loads (depth_simd_dimensions elements per load)
 * - 11× fewer upcasts than naive (32 upcasts for 256 products)
 * - State-based accumulation (supports Neumaier compensation, platform-specific precision)
 * - Parallelization support via row_start/row_count parameters
 */
#define nk_define_dots_symmetric_(name, suffix, input_type, output_type, vec_type, state_type, result_vec_type,        \
                                  init_fn, load_fn, partial_load_fn, update_fn, finalize_fn, depth_simd_dimensions,    \
                                  dimensions_per_value)                                                                \
    NK_PUBLIC void nk_dots_symmetric_##name##_##suffix(                                                                \
        nk_##input_type##_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,                    \
        nk_##output_type##_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {             \
                                                                                                                       \
        /* Const expressions for tiling (following dots_packed pattern) */                                             \
        nk_size_t const register_tile_size = 16;  /* 16×16 register tiles */                                           \
        nk_size_t const finalizer_batch_size = 4; /* Finalizer processes 4 at a time */                                \
                                                                                                                       \
        nk_size_t const vectors_stride_elements = stride / sizeof(nk_##input_type##_t);                                \
        nk_size_t const result_stride_elements = result_stride / sizeof(nk_##output_type##_t);                         \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
                                                                                                                       \
        /* Clamp row range to valid bounds */                                                                          \
        nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;           \
                                                                                                                       \
        /* Process 16×16 tiles of the upper triangle */                                                                \
        for (nk_size_t i_tile = row_start; i_tile < row_end; i_tile += register_tile_size) {                           \
            nk_size_t tile_i_size = ((i_tile + register_tile_size) <= n_vectors) ? register_tile_size                  \
                                                                                 : (n_vectors - i_tile);               \
                                                                                                                       \
            for (nk_size_t j_tile = i_tile; j_tile < n_vectors; j_tile += register_tile_size) {                        \
                nk_size_t tile_j_size = ((j_tile + register_tile_size) <= n_vectors) ? register_tile_size              \
                                                                                     : (n_vectors - j_tile);           \
                                                                                                                       \
                /* Initialize 16×16 accumulator tile (stack → L1, ~4 KB for f32) */                                    \
                state_type accumulator_tiles[16][16];                                                                  \
                for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                       \
                    for (nk_size_t jj = 0; jj < tile_j_size; jj++) { init_fn(&accumulator_tiles[ii][jj]); }            \
                }                                                                                                      \
                                                                                                                       \
                /* Setup row pointers (matches dots_packed pattern) */                                                 \
                nk_##input_type##_t const *row_i[16];                                                                  \
                for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                       \
                    row_i[ii] = vectors + (i_tile + ii) * vectors_stride_elements;                                     \
                }                                                                                                      \
                nk_##input_type##_t const *row_j[16];                                                                  \
                for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                                       \
                    row_j[jj] = vectors + (j_tile + jj) * vectors_stride_elements;                                     \
                }                                                                                                      \
                                                                                                                       \
                /* Tight depth loop (matches dots_packed pattern) */                                                   \
                vec_type vec_i[16];                                                                                    \
                vec_type vec_j[16];                                                                                    \
                for (nk_size_t d = 0; d < aligned_depth; d += depth_step_values) {                                     \
                    /* Load 16 i-vectors (load_fn does upcast) */                                                      \
                    for (nk_size_t ii = 0; ii < tile_i_size; ii++) { load_fn(row_i[ii] + d, &vec_i[ii]); }             \
                    /* Load 16 j-vectors (load_fn does upcast) */                                                      \
                    for (nk_size_t jj = 0; jj < tile_j_size; jj++) { load_fn(row_j[jj] + d, &vec_j[jj]); }             \
                    /* 256 FMAs: 16 i-rows × 16 j-columns */                                                           \
                    for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                   \
                        for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                               \
                            update_fn(&accumulator_tiles[ii][jj], vec_i[ii], vec_j[jj], d * dimensions_per_value,      \
                                      depth_simd_dimensions);                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Handle remainder */                                                                                 \
                if (remainder_depth > 0) {                                                                             \
                    for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                   \
                        partial_load_fn(row_i[ii] + aligned_depth, &vec_i[ii], remainder_dimensions);                  \
                    }                                                                                                  \
                    for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                                   \
                        partial_load_fn(row_j[jj] + aligned_depth, &vec_j[jj], remainder_dimensions);                  \
                    }                                                                                                  \
                    for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                   \
                        for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                               \
                            update_fn(&accumulator_tiles[ii][jj], vec_i[ii], vec_j[jj],                                \
                                      aligned_depth * dimensions_per_value, remainder_dimensions);                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Finalize and store (process 4 columns at a time per finalizer) */                                   \
                for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                       \
                    for (nk_size_t jj = 0; jj < tile_j_size; jj += finalizer_batch_size) {                             \
                        nk_size_t batch_size = ((jj + finalizer_batch_size) <= tile_j_size) ? finalizer_batch_size     \
                                                                                            : (tile_j_size - jj);      \
                                                                                                                       \
                        state_type dummy_b, dummy_c, dummy_d;                                                          \
                        init_fn(&dummy_b);                                                                             \
                        init_fn(&dummy_c);                                                                             \
                        init_fn(&dummy_d);                                                                             \
                                                                                                                       \
                        result_vec_type result_vec;                                                                    \
                        finalize_fn(&accumulator_tiles[ii][jj],                                                        \
                                    (batch_size > 1) ? &accumulator_tiles[ii][jj + 1] : &dummy_b,                      \
                                    (batch_size > 2) ? &accumulator_tiles[ii][jj + 2] : &dummy_c,                      \
                                    (batch_size > 3) ? &accumulator_tiles[ii][jj + 3] : &dummy_d, &result_vec, depth); \
                                                                                                                       \
                        /* Store and mirror */                                                                         \
                        for (nk_size_t col = 0; col < batch_size; col++) {                                             \
                            nk_##output_type##_t val = result_vec.output_type##s[col];                                 \
                            result[(i_tile + ii) * result_stride_elements + (j_tile + jj + col)] = val;                \
                            if (i_tile + ii != j_tile + jj + col) {                                                    \
                                result[(j_tile + jj + col) * result_stride_elements + (i_tile + ii)] = val;            \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes) */
nk_define_dots_pack_size_(f64, serial, f64, f64, f64, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f64, serial, f64, f64, f64, nk_assign_from_to_, /*depth_simd_dimensions=*/2,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f64, serial, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                          nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                          nk_dot_f64x2_update_serial, nk_dot_f64x2_finalize_serial, /*depth_simd_dimensions=*/2,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(f64, serial, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                       nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                       nk_load_b128_serial_, nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_serial,
                       nk_dot_f64x2_finalize_serial, nk_partial_store_b64x4_serial_,
                       /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes) */
nk_define_dots_pack_size_(f32, serial, f32, f32, f32, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f32, serial, f32, f32, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/4,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f32, serial, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                          nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                          nk_dot_f32x4_update_serial, nk_dot_f32x4_finalize_serial, /*depth_simd_dimensions=*/4,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(f32, serial, f32, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                       nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                       nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_serial,
                       nk_dot_f32x4_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(f16, serial, f16, f32, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f16, serial, f16, f16, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f16, serial, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                          nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                          nk_dot_f16x8_update_serial, nk_dot_f16x8_finalize_serial, /*depth_simd_dimensions=*/8,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(f16, serial, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                       nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                       nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_serial,
                       nk_dot_f16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(bf16, serial, bf16, f32, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(bf16, serial, bf16, bf16, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(bf16, serial, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                          nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                          nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial, /*depth_simd_dimensions=*/8,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(bf16, serial, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                       nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                       nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_serial,
                       nk_dot_bf16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes), I32 accumulator */
nk_define_dots_pack_size_(i8, serial, i8, i8, i32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(i8, serial, i8, i8, i32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(i8, serial, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_i8x16_update_serial, nk_dot_i8x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(i8, serial, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_serial,
                       nk_dot_i8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes), U32 accumulator */
nk_define_dots_pack_size_(u8, serial, u8, u8, u32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(u8, serial, u8, u8, u32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(u8, serial, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(u8, serial, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_serial,
                       nk_dot_u8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=16 (16 e4m3s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(e4m3, serial, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e4m3, serial, e4m3, e4m3, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e4m3, serial, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e4m3x16_update_serial, nk_dot_e4m3x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(e4m3, serial, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_serial,
                       nk_dot_e4m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=16 (16 e5m2s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(e5m2, serial, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e5m2, serial, e5m2, e5m2, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e5m2, serial, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e5m2x16_update_serial, nk_dot_e5m2x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(e5m2, serial, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_serial,
                       nk_dot_e5m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=16 (16 e2m3s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(e2m3, serial, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e2m3, serial, e2m3, e2m3, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e2m3, serial, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e2m3x16_update_serial, nk_dot_e2m3x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(e2m3, serial, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_serial,
                       nk_dot_e2m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=16 (16 e3m2s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(e3m2, serial, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e3m2, serial, e3m2, e3m2, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e3m2, serial, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e3m2x16_update_serial, nk_dot_e3m2x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/1)
nk_define_dots_packed_(e3m2, serial, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_serial,
                       nk_dot_e3m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U4 GEMM: u4x2 for both A and B */
nk_define_dots_pack_size_(u4, serial, u4x2, u4x2, u32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_dots_pack_(u4, serial, u4x2, u4x2, u32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/2)
nk_define_dots_symmetric_(u4, serial, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                          nk_dot_u4x16_update_serial, nk_dot_u4x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/2)
nk_define_dots_packed_(u4, serial, u4x2, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                       nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_u4x16_update_serial,
                       nk_dot_u4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* I4 GEMM: i4x2 for both A and B */
nk_define_dots_pack_size_(i4, serial, i4x2, i4x2, i32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_dots_pack_(i4, serial, i4x2, i4x2, i32, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/2)
nk_define_dots_symmetric_(i4, serial, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                          nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial, /*depth_simd_dimensions=*/16,
                          /*dimensions_per_value=*/2)
nk_define_dots_packed_(i4, serial, i4x2, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                       nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_i4x16_update_serial,
                       nk_dot_i4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride_in_bytes, writes BF16 tightly packed (stride = column_count × sizeof(bf16)).
 */
NK_PUBLIC void nk_dots_compact_bf16_serial(void *c, nk_size_t row_count, nk_size_t column_count,
                                           nk_size_t c_stride_in_bytes) {
    nk_size_t const c_stride_in_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_f32_t const *source_row = c_f32 + row_index * c_stride_in_elements;
        nk_bf16_t *destination_row = c_bf16 + row_index * column_count;
        for (nk_size_t column_index = 0; column_index < column_count; column_index++) {
            nk_f32_to_bf16_serial(source_row + column_index, destination_row + column_index);
        }
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] × 127 / sqrt(a_norm[i] × b_norm[j])
 *  Output is tightly packed (stride_in_bytes = column_count × sizeof(i8)).
 */
NK_PUBLIC void nk_dots_compact_i8_serial(void *c, nk_size_t row_count, nk_size_t column_count,
                                         nk_size_t c_stride_in_bytes, nk_i32_t const *a_squared_norms,
                                         nk_i32_t const *b_squared_norms) {
    nk_size_t const c_stride_in_elements = c_stride_in_bytes / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_i32_t const *source_row = c_i32 + row_index * c_stride_in_elements;
        nk_i8_t *destination_row = c_i8 + row_index * column_count;

        nk_f32_t a_norm_f32_value = (nk_f32_t)a_squared_norms[row_index];
        nk_f32_t a_rsqrt_value = (a_norm_f32_value > 0) ? (1.0f / nk_f32_sqrt_serial(a_norm_f32_value)) : 0.0f;

        for (nk_size_t column_index = 0; column_index < column_count; column_index++) {
            nk_f32_t b_norm_f32_value = (nk_f32_t)b_squared_norms[column_index];
            nk_f32_t b_rsqrt_value = (b_norm_f32_value > 0) ? (1.0f / nk_f32_sqrt_serial(b_norm_f32_value)) : 0.0f;

            nk_f32_t normalized_value = (nk_f32_t)source_row[column_index] * 127.0f * a_rsqrt_value * b_rsqrt_value;
            nk_i32_t clamped_value = (nk_i32_t)normalized_value;
            if (clamped_value < -128) clamped_value = -128;
            if (clamped_value > 127) clamped_value = 127;
            destination_row[column_index] = (nk_i8_t)clamped_value;
        }
    }
}

#if defined(__cplusplus)
}
#endif

#endif // NK_DOTS_SERIAL_H
