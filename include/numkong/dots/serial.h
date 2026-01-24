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
    nk_u32_t depth_dimensions;    // Logical depth in dimensions (nibbles for i4/u4, values for i8/f32)
    nk_u32_t depth_padded_values; // Padded depth in storage values (bytes for i4/u4, values for i8/f32)
    nk_u32_t reserved[13];        // Padding to 64 bytes
} nk_cross_packed_buffer_header_t;

/**
 *  @brief Generates function to calculate packed B matrix buffer size for GEMM micro-kernels.
 *
 *  Memory layout: B_packed[column_count, depth_padded] with header storing metadata.
 *  Buffer size: sizeof(header) + column_count × depth_padded × sizeof(intermediate_type)
 *  Depth padding logic: Round up to `depth_simd_dimensions` multiple, then add `depth_simd_dimensions`
 *  if stride is power-of-2.
 *
 *  @param api_name Operation name (hammings, dots)
 *  @param input_type_name Original type's name of B matrix values (i4, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param isa_suffix Platform Instruct Set Architecture suffix (serial, haswell, ice, etc.)
 *  @param input_type Original type of B matrix values (i4x2, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param intermediate_type Internal storage type in packed buffer (often bf16 or f32 for mixed precision)
 *  @param depth_simd_dimensions SIMD vector width in values for this platform/type combination
 *  @param dimensions_per_value Number of logical dimensions in a single value of input_type_name.
 */
#define nk_define_cross_pack_size_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,   \
                                   depth_simd_dimensions, dimensions_per_value)                                  \
    NK_PUBLIC nk_size_t nk_##api_name##_packed_size_##input_type_name##_##isa_suffix(nk_size_t column_count,     \
                                                                                     nk_size_t depth) {          \
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
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                 \
                                                                                                                 \
        /* Step 4: Break power-of-2 strides for cache associativity */                                           \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                      \
            /* Add one SIMD step worth of storage values */                                                      \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);        \
        }                                                                                                        \
                                                                                                                 \
        /* Step 5: Return total buffer size in bytes */                                                          \
        return sizeof(nk_cross_packed_buffer_header_t) +                                                         \
               column_count * depth_values_padded * sizeof(nk_##packed_value_type##_t);                          \
    }

/**
 *  @brief Generates function to pack and optionally convert B matrix for efficient GEMM inner loops.
 *
 *  Packing serves two performance-critical purposes:
 *
 *  1. Type conversion (input_type → intermediate_type): For mixed-precision GEMM, convert B values
 *     once during packing rather than repeatedly in tight inner loops. Example: F16 → F32 conversion
 *     happens once per value instead of once per (row of A × value of B) access. This amortizes
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
 *  @param api_name Operation name (hammings, dots)
 *  @param input_type_name Original type's name of B matrix values (i4, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param isa_suffix Platform Instruct Set Architecture suffix (serial, haswell, ice, etc.)
 *  @param input_type Original type of B matrix values (i4x2, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param intermediate_type Internal storage type in packed buffer (often bf16 or f32 for mixed precision)
 *  @param convert_value_fn Element conversion function: void fn(input_type const*, intermediate_type*)
 *  @param depth_simd_dimensions SIMD vector width in values for depth padding alignment
 *  @param dimensions_per_value Number of logical dimensions in a single value of input_type.
 */
#define nk_define_cross_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,             \
                              convert_value_fn, depth_simd_dimensions, dimensions_per_value)                          \
    NK_PUBLIC void nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                             \
        nk_##input_value_type##_t const *b, nk_size_t column_count, nk_size_t depth, nk_size_t b_stride_in_bytes,     \
        void *b_packed) {                                                                                             \
        /* Use identical padding calculation as pack_size */                                                          \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);              \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value);      \
                                                                                                                      \
        /* Power-of-2 breaking (same as pack_size) */                                                                 \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                      \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                           \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);             \
        }                                                                                                             \
                                                                                                                      \
        /* Calculate input depth in values */                                                                         \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                      \
                                                                                                                      \
        /* Store dimensions in header */                                                                              \
        nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                        \
        header->column_count = (nk_u32_t)column_count;                                                                \
        header->depth_dimensions = (nk_u32_t)depth;                  /* depth in dimensions (nibbles for i4/u4) */    \
        header->depth_padded_values = (nk_u32_t)depth_values_padded; /* padded depth in VALUES (bytes for i4/u4) */   \
                                                                                                                      \
        nk_##packed_value_type##_t *packed = (nk_##packed_value_type##_t *)((char *)b_packed +                        \
                                                                            sizeof(nk_cross_packed_buffer_header_t)); \
                                                                                                                      \
        /* Zero entire buffer for depth padding */                                                                    \
        nk_size_t const total_values = column_count * depth_values_padded;                                            \
        for (nk_size_t i = 0; i < total_values; ++i) packed[i] = 0;                                                   \
                                                                                                                      \
        /* Copy/convert B[column_count, depth] to packed[column_count, depth_padded] - simple column-major */         \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                               \
            nk_##packed_value_type##_t *destination_row = packed + column_index * depth_values_padded;                \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride_in_bytes);              \
            for (nk_size_t depth_index = 0; depth_index < depth_in_values; ++depth_index) {                           \
                convert_value_fn(&source_row[depth_index], &destination_row[depth_index]);                            \
            }                                                                                                         \
            /* Padding values already zeroed above */                                                                 \
        }                                                                                                             \
    }

/**
 *  @brief Generates optimized GEMM implementation: C = A × Bᵀ with pre-packed B matrix.
 *
 *  This macro creates a complete batched matrix multiplication kernel with THREE specialized
 *  code paths that are automatically selected based on the remaining work at each blocking level.
 *  The kernel requires B to be pre-packed using nk_define_cross_pack_ before invocation.
 *
 *  **Mathematical operation**:
 *    C[row_count, column_count] = A[row_count, depth] × Bᵀ[column_count, depth]
 *  where operation can be dot product, Hamming distance, Jaccard similarity, etc.
 *
 *  **Three kernel variants for adaptive performance**:
 *
 *  1. **4×4 register tile kernel** (primary path, ~80% of work):
 *     - Processes 4 rows of A × 4 columns of B simultaneously
 *     - Maintains 16 independent accumulators in registers (state_type[4][4])
 *     - Achieves maximum instruction-level parallelism (16 FMAs per depth iteration)
 *     - Used when: row_count ≥ 4 AND column_count ≥ 4
 *     - Performance: Peak throughput, optimal register utilization
 *
 *  2. **1×8 register tile kernel** (edge case, ~15% of work):
 *     - Processes 1 row of A × 8 columns of B when remaining rows < 4
 *     - Maintains 8 independent accumulators (state_type[1][8])
 *     - Balances vectorization with low row count
 *     - Used when: row_count < 4 AND column_count ≥ 8
 *     - Performance: Better throughput than generic fallback for wide matrices
 *
 *  3. **Generic fallback kernel** (edge cases, ~5% of work):
 *     - Handles all irregular cases (row_count < 4 AND column_count < 8)
 *     - Single accumulator, minimal unrolling
 *     - Used for: Small tiles, remainder handling
 *     - Performance: Lower throughput but handles all edge cases correctly
 *
 *  **Cache blocking strategy (no depth blocking)**:
 *
 *  Unlike traditional GEMM which blocks all three dimensions (M, N, K), this implementation
 *  deliberately omits depth (K) blocking for several reasons:
 *
 *  1. **Streaming access pattern**: A and B are read sequentially along depth dimension
 *     - Prefetcher-friendly access (hardware prefetch works well)
 *     - No cache reuse along depth within a single C[i,j] computation
 *
 *  2. **Depth is typically small**: For ML inference, depth is often 128-4096 values
 *     - Fits in L2/L3 cache for single row of A
 *     - B is pre-packed for optimal spatial locality
 *
 *  3. **Simplicity and instruction cache efficiency**:
 *     - Fewer nested loops = better instruction cache utilization
 *     - Simpler control flow = easier for compiler to optimize
 *
 *  **Pre-packing benefits**:
 *
 *  B matrix is pre-packed using nk_define_cross_pack_ before kernel invocation:
 *  - **Type conversion amortization**: Convert B values once (e.g., bf16→f32) rather than
 *    per A row access. Saves (row_count - 1) × column_count conversions.
 *  - **Cache line optimization**: Pad depth to break power-of-2 strides that cause cache
 *    associativity conflicts (e.g., 8192 → 8200 values).
 *  - **Spatial locality**: Transpose B so columns are contiguous, enabling efficient SIMD loads.
 *
 *  **Loop structure**:
 *
 *    for column_block in columns (step: varies based on available columns):
 *      for row_block in rows (step: varies based on available rows):
 *        for row_tile in row_block (step: 4 or 1 depending on variant):
 *          for column_tile in column_block (step: 4 or 8 depending on variant):
 *            accumulator_tiles[row_tile][column_tile] = init_accumulator_fn()
 *            for depth_index in depth (step: depth_simd_dimensions):
 *              a_vectors = load_a_vec_fn(A[row_tile, depth_index])
 *              b_vectors = load_b_vec_fn(B_packed[column_tile, depth_index])
 *              accumulator_tiles = inner_product_fn(accumulator_tiles, a_vectors, b_vectors)
 *            results = reduce_accumulators_fn(accumulator_tiles)
 *            partial_store_fn(results, C[row_tile, column_tile])
 *
 *  **Generated function**:
 *
 *  nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(
 *      A_matrix, B_packed_buffer, C_matrix, row_count, column_count, depth,
 *      A_stride_bytes, C_stride_bytes)
 *
 *  @param api_name Operation family (dots, hammings, jaccards) for codegen namespace
 *  @param input_type_name Type identifier for codegen (f32, bf16, i8, u1, etc.)
 *  @param isa_suffix ISA backend identifier (serial, haswell, neon, sve, ice, etc.)
 *  @param input_type C type of input matrix values (f32, bf16, i8, u1x8, etc.)
 *  @param intermediate_type Storage type in packed B buffer (often bf16 or f32 for mixed precision)
 *  @param output_type C type of output matrix C values (f32, u32, f64, etc.)
 *  @param vec_type SIMD vector type for depth dimension (e.g., __m256, nk_f32x8_t)
 *  @param state_type Accumulator state type (often vec_type or wider, e.g., __m256 or __m512)
 *  @param result_vec_type SIMD vector type for reduction results (e.g., __m128 for 4 f32 results)
 *  @param init_accumulator_fn Initialize accumulator: void fn(state_type*)
 *  @param load_a_vec_fn Full A vector load: vec_type fn(input_type const*, nk_size_t offset)
 *  @param partial_load_a_vec_fn Partial A load for remainder
 *  @param load_b_vec_fn Full B vector load: vec_type fn(intermediate_type const*, nk_size_t offset)
 *  @param partial_load_b_vec_fn Partial B load for remainder
 *  @param inner_product_fn Inner product accumulate
 *  @param reduce_accumulators_fn Reduce 4 accumulators
 *  @param partial_store_fn Partial store for results
 *  @param depth_simd_dimensions SIMD vector width in logical dimensions (e.g., 8 for f32 on AVX2, 128 for u1 on serial)
 *  @param dimensions_per_value Packing ratio: dimensions per storage value (1 for f32, 128 for u1x8)
 *
 *  @sa nk_define_cross_symmetric_ for symmetric C = A × Aᵀ computation (upper triangle only)
 *  @sa nk_define_cross_pack_size_ for calculating B_packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix into optimized layout
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_packed_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,            \
                                result_value_type, vec_type, state_type, result_vec_type, init_accumulator_fn,         \
                                load_a_vec_fn, partial_load_a_vec_fn, load_b_vec_fn, partial_load_b_vec_fn,            \
                                inner_product_fn, reduce_accumulators_fn, partial_store_fn, depth_simd_dimensions,     \
                                dimensions_per_value)                                                                  \
    NK_PUBLIC void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                                 \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
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
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
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
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Register tiling over rows (register_row_count rows per tile) */                         \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                                                                                                                       \
                        /* Initialize register_row_count × register_column_count accumulator states */                 \
                        state_type accumulator_tiles[4][4];                                                            \
                        init_accumulator_fn(&accumulator_tiles[0][0]), init_accumulator_fn(&accumulator_tiles[0][1]),  \
                            init_accumulator_fn(&accumulator_tiles[0][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[0][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[1][0]), init_accumulator_fn(&accumulator_tiles[1][1]),  \
                            init_accumulator_fn(&accumulator_tiles[1][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[1][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[2][0]), init_accumulator_fn(&accumulator_tiles[2][1]),  \
                            init_accumulator_fn(&accumulator_tiles[2][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[2][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[3][0]), init_accumulator_fn(&accumulator_tiles[3][1]),  \
                            init_accumulator_fn(&accumulator_tiles[3][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[3][3]);                                             \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 1) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 2) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 3) * a_stride_in_bytes);       \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                                       \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                                       \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load next few values from 4 rows from A (unpacked, may upcast) */                       \
                            load_a_vec_fn(a_row_ptr_0 + depth_index, &a_vector_0);                                     \
                            load_a_vec_fn(a_row_ptr_1 + depth_index, &a_vector_1);                                     \
                            load_a_vec_fn(a_row_ptr_2 + depth_index, &a_vector_2);                                     \
                            load_a_vec_fn(a_row_ptr_3 + depth_index, &a_vector_3);                                     \
                                                                                                                       \
                            /* Load next few values from 4 rows from B (packed, already upcasted) */                   \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_vector_0, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][1], a_vector_0, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][2], a_vector_0, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][3], a_vector_0, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][0], a_vector_1, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][1], a_vector_1, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][2], a_vector_1, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][3], a_vector_1, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][0], a_vector_2, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][1], a_vector_2, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][2], a_vector_2, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][3], a_vector_2, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][0], a_vector_3, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][1], a_vector_3, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][2], a_vector_3, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][3], a_vector_3, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                        }                                                                                              \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr_0 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 0) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[0][0], &accumulator_tiles[0][1],                     \
                                               &accumulator_tiles[0][2], &accumulator_tiles[0][3], depth,              \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr_0 + tile_column_start_index, 4);                    \
                        nk_##result_value_type##_t *c_row_ptr_1 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 1) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[1][0], &accumulator_tiles[1][1],                     \
                                               &accumulator_tiles[1][2], &accumulator_tiles[1][3], depth,              \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr_1 + tile_column_start_index, 4);                    \
                        nk_##result_value_type##_t *c_row_ptr_2 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 2) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[2][0], &accumulator_tiles[2][1],                     \
                                               &accumulator_tiles[2][2], &accumulator_tiles[2][3], depth,              \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr_2 + tile_column_start_index, 4);                    \
                        nk_##result_value_type##_t *c_row_ptr_3 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 3) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[3][0], &accumulator_tiles[3][1],                     \
                                               &accumulator_tiles[3][2], &accumulator_tiles[3][3], depth,              \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr_3 + tile_column_start_index, 4);                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                             \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values; /* in storage values */                            \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
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
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
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
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_4 = packed_data +                                    \
                                                                      (tile_column_start_index + 4) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_5 = packed_data +                                    \
                                                                      (tile_column_start_index + 5) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_6 = packed_data +                                    \
                                                                      (tile_column_start_index + 6) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_7 = packed_data +                                    \
                                                                      (tile_column_start_index + 7) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Process 1 row at a time */                                                              \
                    for (nk_size_t row_index = row_block_start_index; row_index < row_block_end_index; ++row_index) {  \
                                                                                                                       \
                        /* Initialize 1 × 8 accumulator states */                                                      \
                        state_type accumulator_0, accumulator_1, accumulator_2, accumulator_3, accumulator_4,          \
                            accumulator_5, accumulator_6, accumulator_7;                                               \
                        init_accumulator_fn(&accumulator_0), init_accumulator_fn(&accumulator_1),                      \
                            init_accumulator_fn(&accumulator_2), init_accumulator_fn(&accumulator_3),                  \
                            init_accumulator_fn(&accumulator_4), init_accumulator_fn(&accumulator_5),                  \
                            init_accumulator_fn(&accumulator_6), init_accumulator_fn(&accumulator_7);                  \
                                                                                                                       \
                        /* A row pointer */                                                                            \
                        nk_##input_value_type##_t const *a_row_ptr =                                                   \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                row_index * a_stride_in_bytes);                        \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector;                                                                             \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3, b_vector_4, b_vector_5, b_vector_6,   \
                            b_vector_7;                                                                                \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load A vector (1 row) */                                                                \
                            load_a_vec_fn(a_row_ptr + depth_index, &a_vector);                                         \
                                                                                                                       \
                            /* Load B vectors (8 columns) */                                                           \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                            load_b_vec_fn(b_depth_ptr_4 + depth_index, &b_vector_4);                                   \
                            load_b_vec_fn(b_depth_ptr_5 + depth_index, &b_vector_5);                                   \
                            load_b_vec_fn(b_depth_ptr_6 + depth_index, &b_vector_6);                                   \
                            load_b_vec_fn(b_depth_ptr_7 + depth_index, &b_vector_7);                                   \
                                                                                                                       \
                            /* 8 FMAs: 1 A row × 8 B columns */                                                        \
                            inner_product_fn(&accumulator_0, a_vector, b_vector_0, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_1, a_vector, b_vector_1, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_2, a_vector, b_vector_2, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_3, a_vector, b_vector_3, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_4, a_vector, b_vector_4, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_5, a_vector, b_vector_5, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_6, a_vector, b_vector_6, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_7, a_vector, b_vector_7, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store 1 × 8 results using two 4-way reductions */                              \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr =                                                        \
                            (nk_##result_value_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);          \
                        /* First 4 columns */                                                                          \
                        reduce_accumulators_fn(&accumulator_0, &accumulator_1, &accumulator_2, &accumulator_3, depth,  \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index, 4);                      \
                        /* Second 4 columns */                                                                         \
                        reduce_accumulators_fn(&accumulator_4, &accumulator_5, &accumulator_6, &accumulator_7, depth,  \
                                               &result_vector);                                                        \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index + 4, 4);                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                            \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
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
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                                    \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride_in_bytes,                \
                c_stride_in_bytes);                                                                                    \
            return;                                                                                                    \
        }                                                                                                              \
        /* Use 4 × 4 kernel when dimensions are 4-aligned */                                                           \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(a_matrix, b_packed_buffer, c_matrix,    \
                                                                               row_count, column_count, depth,         \
                                                                               a_stride_in_bytes, c_stride_in_bytes);  \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
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
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
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
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 =                                                  \
                        (tile_column_count > 1) ? packed_data + (tile_column_start_index + 1) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 =                                                  \
                        (tile_column_count > 2) ? packed_data + (tile_column_start_index + 2) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 =                                                  \
                        (tile_column_count > 3) ? packed_data + (tile_column_start_index + 3) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
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
                            init_accumulator_fn(&accumulator_tiles[r][0]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][1]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][2]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][3]);                                             \
                        }                                                                                              \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (tile_row_count > 1)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 1) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (tile_row_count > 2)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 2) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (tile_row_count > 3)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 3) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                                                                                                                       \
                        /* Tight inner loop: k values with simple ptr+k addressing */                                  \
                        vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                                 \
                        vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                                 \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            /* Load next few values from 4 rows from A */                                              \
                            load_a_vec_fn(a_row_ptr_0 + k, &a_first_vec);                                              \
                            load_a_vec_fn(a_row_ptr_1 + k, &a_second_vec);                                             \
                            load_a_vec_fn(a_row_ptr_2 + k, &a_third_vec);                                              \
                            load_a_vec_fn(a_row_ptr_3 + k, &a_fourth_vec);                                             \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            load_b_vec_fn(b_depth_ptr_0 + k, &b_first_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_1 + k, &b_second_vec);                                           \
                            load_b_vec_fn(b_depth_ptr_2 + k, &b_third_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_3 + k, &b_fourth_vec);                                           \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                        }                                                                                              \
                                                                                                                       \
                        /* Handle remainder k positions with partial loads */                                          \
                        if (remainder_depth > 0) {                                                                     \
                            /* Load next few values from 4 rows from A */                                              \
                            partial_load_a_vec_fn(a_row_ptr_0 + aligned_depth, &a_first_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_1 + aligned_depth, &a_second_vec, remainder_dimensions);   \
                            partial_load_a_vec_fn(a_row_ptr_2 + aligned_depth, &a_third_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_3 + aligned_depth, &a_fourth_vec, remainder_dimensions);   \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            partial_load_b_vec_fn(b_depth_ptr_0 + aligned_depth, &b_first_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_1 + aligned_depth, &b_second_vec, remainder_dimensions); \
                            partial_load_b_vec_fn(b_depth_ptr_2 + aligned_depth, &b_third_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_3 + aligned_depth, &b_fourth_vec, remainder_dimensions); \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            result_vec_type result_vector;                                                             \
                            reduce_accumulators_fn(&accumulator_tiles[r][0], &accumulator_tiles[r][1],                 \
                                                   &accumulator_tiles[r][2], &accumulator_tiles[r][3], depth,          \
                                                   &result_vector);                                                    \
                                                                                                                       \
                            nk_##result_value_type##_t *c_row =                                                        \
                                (nk_##result_value_type##_t *)((char *)c_matrix +                                      \
                                                               (tile_row_start_index + r) * c_stride_in_bytes);        \
                            partial_store_fn(&result_vector, c_row + tile_column_start_index, tile_column_count);      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Generates optimized symmetric Gram matrix computation: C = A × Aᵀ (upper triangle only).
 *
 *  This macro creates a complete symmetric cross-product implementation with two specialized
 *  internal helper functions (diagonal and off-diagonal) that are called by a public wrapper.
 *  Symmetric computation exploits the property that C[i,j] = C[j,i], computing only the upper
 *  triangle and avoiding redundant computation and storage.
 *
 *  **Mathematical operation**: For each pair (i,j) where i ≤ j:
 *    C[i,j] = operation(A[i,:], A[j,:])
 *  where operation can be dot product, Hamming distance, Jaccard similarity, etc.
 *
 *  **Architecture - Three-level tiling hierarchy**:
 *
 *  1. **32×32 macro-tiles** (outermost): Divides the upper triangle into 32×32 blocks
 *     - Rationale: Fits well in L1 cache (32 vectors × depth × value_size)
 *     - Enables diagonal vs off-diagonal specialization
 *     - Amortizes vector loads across all depth iterations
 *     - Pre-loads and upcasts ALL 32 vectors ONCE per depth iteration (not per FMA)
 *
 *  2. **4×4 register tiles** (middle): Within each macro-tile, process 4×4 sub-blocks
 *     - Rationale: Maximizes register reuse (4 A vectors × 4 A vectors = 16 accumulators)
 *     - Enables full FMA unrolling (16 FMAs for off-diagonal, 10 for diagonal)
 *     - Balances register pressure with instruction-level parallelism
 *
 *  3. **Depth loop** (innermost): For each depth chunk, accumulate outer products
 *     - Depth loop is INSIDE macro-tile, OUTSIDE register tiles
 *     - Type conversion (e.g., bf16→f32) happens at macro-tile level (once per vector)
 *
 *  **Diagonal vs off-diagonal optimization**:
 *
 *  - **Diagonal macro-tiles** (i_macro == j_macro): Computes C[i:i+32, i:i+32]
 *    - Loads 32 vectors ONCE (50% load reduction vs off-diagonal)
 *    - Computes upper triangle only within the tile (10 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  - **Off-diagonal macro-tiles** (i_macro < j_macro): Computes C[i:i+32, j:j+32]
 *    - Loads vec_i[32] + vec_j[32] (full 64 vectors for two sets)
 *    - Computes full 32×32 block (16 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  **When to use symmetric vs packed variant**:
 *
 *  - Use symmetric (this macro) when: A is the SAME matrix for both sides (C = A × Aᵀ)
 *    - Saves 50% computation and storage (upper triangle only)
 *    - Automatic diagonal optimization (50% fewer loads on diagonal tiles)
 *    - Ideal for: distance matrices, correlation matrices, Gram matrices
 *
 *  - Use packed variant when: Computing C = A × Bᵀ where A ≠ B
 *    - Full matrix computation (no symmetry to exploit)
 *    - B can be pre-packed for cache efficiency
 *
 *  **Generated functions**:
 *
 *  This macro generates THREE functions:
 *  1. nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_ (NK_INTERNAL)
 *  2. nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_ (NK_INTERNAL)
 *  3. nk_##api_name##_symmetric_##input_type_name##_##isa_suffix (NK_PUBLIC wrapper)
 *
 *  @param api_name Operation family (dots, hammings, jaccards) for codegen namespace
 *  @param input_type_name Type identifier for codegen (f32, bf16, i8, u1, etc.)
 *  @param isa_suffix ISA backend identifier (serial, haswell, neon, sve, ice, etc.)
 *  @param input_type C type of input matrix values (f32, bf16, i8, u1x8, etc.)
 *  @param output_type C type of output matrix values (f32, u32, f64, etc.)
 *  @param vec_type SIMD vector type for input vectors (e.g., __m256, nk_f32x8_t)
 *  @param state_type Accumulator state type (often vec_type or wider, e.g., __m256 or __m512)
 *  @param result_vec_type SIMD vector type for reduction results (e.g., __m128 for 4 f32 results)
 *  @param init_accumulator_fn Initialize accumulator: void fn(state_type*)
 *  @param load_vec_fn Full vector load: vec_type fn(input_type const*, nk_size_t offset)
 *  @param partial_load_vec_fn Partial vector load for remainder
 *  @param inner_product_fn Inner product accumulate
 *  @param reduce_accumulators_fn Reduce 4 accumulators
 *  @param partial_store_fn Partial store for results
 *  @param depth_simd_dimensions SIMD vector width in logical dimensions (e.g., 8 for f32 on AVX2, 128 for u1 on serial)
 *  @param dimensions_per_value Packing ratio: dimensions per storage value (1 for f32, 128 for u1x8)
 *
 *  @sa nk_define_cross_packed_ for asymmetric C = A × Bᵀ computation
 *  @sa nk_define_cross_pack_size_ for calculating packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_symmetric_(api_name, input_type_name, isa_suffix, input_value_type, result_value_type,         \
                                   vec_type, state_type, result_vec_type, init_accumulator_fn, load_vec_fn,            \
                                   partial_load_vec_fn, inner_product_fn, reduce_accumulators_fn, partial_store_fn,    \
                                   depth_simd_dimensions, dimensions_per_value)                                        \
    NK_INTERNAL void nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                           \
        state_type accumulator_tiles[32][32], nk_##input_value_type##_t const **vector_base_ptrs, nk_size_t i_macro,   \
        nk_size_t macro_size, nk_size_t aligned_depth, nk_size_t remainder_depth, nk_size_t remainder_dimensions,      \
        nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result,       \
        nk_size_t result_stride_values, nk_size_t finalizer_batch_size, nk_size_t depth) {                             \
                                                                                                                       \
        /* Initialize accumulators (upper triangle only) */                                                            \
        for (nk_size_t ii = 0; ii < macro_size; ii++) {                                                                \
            for (nk_size_t jj = ii; jj < macro_size; jj++) { init_accumulator_fn(&accumulator_tiles[ii][jj]); }        \
        }                                                                                                              \
                                                                                                                       \
        /* Aligned depth loop */                                                                                       \
        vec_type vec[32];                                                                                              \
        for (nk_size_t d = 0; d < aligned_depth; d += depth_step_values) {                                             \
            for (nk_size_t i = 0; i < macro_size; i++) { load_vec_fn(vector_base_ptrs[i] + d, &vec[i]); }              \
            nk_size_t offset = d * dimensions_per_value;                                                               \
                                                                                                                       \
            for (nk_size_t i_tile = 0; i_tile < macro_size; i_tile += 4) {                                             \
                for (nk_size_t j_tile = i_tile; j_tile < macro_size; j_tile += 4) {                                    \
                    nk_size_t tile_i_size = (i_tile + 4 <= macro_size) ? 4 : (macro_size - i_tile);                    \
                    nk_size_t tile_j_size = (j_tile + 4 <= macro_size) ? 4 : (macro_size - j_tile);                    \
                                                                                                                       \
                    if (i_tile == j_tile) {                                                                            \
                        if (tile_i_size == 4 && tile_j_size == 4) {                                                    \
                            inner_product_fn(&accumulator_tiles[i_tile + 0][j_tile + 0], vec[i_tile + 0],              \
                                             vec[j_tile + 0], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 0][j_tile + 1], vec[i_tile + 0],              \
                                             vec[j_tile + 1], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 0][j_tile + 2], vec[i_tile + 0],              \
                                             vec[j_tile + 2], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 0][j_tile + 3], vec[i_tile + 0],              \
                                             vec[j_tile + 3], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 1][j_tile + 1], vec[i_tile + 1],              \
                                             vec[j_tile + 1], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 1][j_tile + 2], vec[i_tile + 1],              \
                                             vec[j_tile + 2], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 1][j_tile + 3], vec[i_tile + 1],              \
                                             vec[j_tile + 3], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 2][j_tile + 2], vec[i_tile + 2],              \
                                             vec[j_tile + 2], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 2][j_tile + 3], vec[i_tile + 2],              \
                                             vec[j_tile + 3], offset, depth_simd_dimensions);                          \
                            inner_product_fn(&accumulator_tiles[i_tile + 3][j_tile + 3], vec[i_tile + 3],              \
                                             vec[j_tile + 3], offset, depth_simd_dimensions);                          \
                        }                                                                                              \
                        else {                                                                                         \
                            for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                           \
                                for (nk_size_t jj = ii; jj < tile_j_size; jj++) {                                      \
                                    inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec[i_tile + ii],   \
                                                     vec[j_tile + jj], offset, depth_simd_dimensions);                 \
                                }                                                                                      \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                    else {                                                                                             \
                        if (tile_i_size == 4 && tile_j_size == 4) {                                                    \
                            for (nk_size_t ii = 0; ii < 4; ii++) {                                                     \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 0], vec[i_tile + ii],        \
                                                 vec[j_tile + 0], offset, depth_simd_dimensions);                      \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 1], vec[i_tile + ii],        \
                                                 vec[j_tile + 1], offset, depth_simd_dimensions);                      \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 2], vec[i_tile + ii],        \
                                                 vec[j_tile + 2], offset, depth_simd_dimensions);                      \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 3], vec[i_tile + ii],        \
                                                 vec[j_tile + 3], offset, depth_simd_dimensions);                      \
                            }                                                                                          \
                        }                                                                                              \
                        else {                                                                                         \
                            for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                           \
                                for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                       \
                                    inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec[i_tile + ii],   \
                                                     vec[j_tile + jj], offset, depth_simd_dimensions);                 \
                                }                                                                                      \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (remainder_depth > 0) {                                                                                     \
            for (nk_size_t i = 0; i < macro_size; i++) {                                                               \
                partial_load_vec_fn(vector_base_ptrs[i] + aligned_depth, &vec[i], remainder_dimensions);               \
            }                                                                                                          \
            nk_size_t offset = aligned_depth * dimensions_per_value;                                                   \
            for (nk_size_t i_tile = 0; i_tile < macro_size; i_tile += 4) {                                             \
                for (nk_size_t j_tile = i_tile; j_tile < macro_size; j_tile += 4) {                                    \
                    nk_size_t tile_i_size = (i_tile + 4 <= macro_size) ? 4 : (macro_size - i_tile);                    \
                    nk_size_t tile_j_size = (j_tile + 4 <= macro_size) ? 4 : (macro_size - j_tile);                    \
                    if (i_tile == j_tile) {                                                                            \
                        for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                               \
                            for (nk_size_t jj = ii; jj < tile_j_size; jj++) {                                          \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec[i_tile + ii],       \
                                                 vec[j_tile + jj], offset, remainder_dimensions);                      \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                               \
                            for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                           \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec[i_tile + ii],       \
                                                 vec[j_tile + jj], offset, remainder_dimensions);                      \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        /* Finalize and store (upper triangle only, no mirroring) */                                                   \
        for (nk_size_t ii = 0; ii < macro_size; ii++) {                                                                \
            nk_size_t jj = ii;                                                                                         \
            /* Process 4 at a time */                                                                                  \
            for (; jj + 4 <= macro_size; jj += 4) {                                                                    \
                result_vec_type result_vec;                                                                            \
                reduce_accumulators_fn(&accumulator_tiles[ii][jj], &accumulator_tiles[ii][jj + 1],                     \
                                       &accumulator_tiles[ii][jj + 2], &accumulator_tiles[ii][jj + 3], depth,          \
                                       &result_vec);                                                                   \
                nk_##result_value_type##_t *out_ptr = &result[(i_macro + ii) * result_stride_values + (i_macro + jj)]; \
                partial_store_fn(&result_vec, out_ptr, 4);                                                             \
            }                                                                                                          \
            /* Handle remaining values (< 4) */                                                                        \
            if (jj < macro_size) {                                                                                     \
                nk_size_t remaining = macro_size - jj;                                                                 \
                result_vec_type result_vec;                                                                            \
                reduce_accumulators_fn(&accumulator_tiles[ii][jj], &accumulator_tiles[ii][jj + 1],                     \
                                       &accumulator_tiles[ii][jj + 2], &accumulator_tiles[ii][jj + 3], depth,          \
                                       &result_vec);                                                                   \
                nk_##result_value_type##_t *out_ptr = &result[(i_macro + ii) * result_stride_values + (i_macro + jj)]; \
                partial_store_fn(&result_vec, out_ptr, remaining);                                                     \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_INTERNAL void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal_(                        \
        state_type accumulator_tiles[32][32], nk_##input_value_type##_t const **vector_base_ptrs_i,                    \
        nk_##input_value_type##_t const **vector_base_ptrs_j, nk_size_t i_macro, nk_size_t j_macro,                    \
        nk_size_t macro_i_size, nk_size_t macro_j_size, nk_size_t aligned_depth, nk_size_t remainder_depth,            \
        nk_size_t remainder_dimensions, nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime,           \
        nk_##result_value_type##_t *result, nk_size_t result_stride_values, nk_size_t finalizer_batch_size,            \
        nk_size_t depth) {                                                                                             \
                                                                                                                       \
        /* Initialize accumulators (full rectangle) */                                                                 \
        for (nk_size_t ii = 0; ii < macro_i_size; ii++) {                                                              \
            for (nk_size_t jj = 0; jj < macro_j_size; jj++) { init_accumulator_fn(&accumulator_tiles[ii][jj]); }       \
        }                                                                                                              \
                                                                                                                       \
        /* Aligned depth loop */                                                                                       \
        vec_type vec_i[32];                                                                                            \
        vec_type vec_j[32];                                                                                            \
        for (nk_size_t d = 0; d < aligned_depth; d += depth_step_values) {                                             \
            for (nk_size_t i = 0; i < macro_i_size; i++) { load_vec_fn(vector_base_ptrs_i[i] + d, &vec_i[i]); }        \
            for (nk_size_t j = 0; j < macro_j_size; j++) { load_vec_fn(vector_base_ptrs_j[j] + d, &vec_j[j]); }        \
            nk_size_t offset = d * dimensions_per_value;                                                               \
                                                                                                                       \
            for (nk_size_t i_tile = 0; i_tile < macro_i_size; i_tile += 4) {                                           \
                for (nk_size_t j_tile = 0; j_tile < macro_j_size; j_tile += 4) {                                       \
                    nk_size_t tile_i_size = (i_tile + 4 <= macro_i_size) ? 4 : (macro_i_size - i_tile);                \
                    nk_size_t tile_j_size = (j_tile + 4 <= macro_j_size) ? 4 : (macro_j_size - j_tile);                \
                    if (tile_i_size == 4 && tile_j_size == 4) {                                                        \
                        for (nk_size_t ii = 0; ii < 4; ii++) {                                                         \
                            inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 0], vec_i[i_tile + ii],          \
                                             vec_j[j_tile + 0], offset, depth_simd_dimensions);                        \
                            inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 1], vec_i[i_tile + ii],          \
                                             vec_j[j_tile + 1], offset, depth_simd_dimensions);                        \
                            inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 2], vec_i[i_tile + ii],          \
                                             vec_j[j_tile + 2], offset, depth_simd_dimensions);                        \
                            inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + 3], vec_i[i_tile + ii],          \
                                             vec_j[j_tile + 3], offset, depth_simd_dimensions);                        \
                        }                                                                                              \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                               \
                            for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                           \
                                inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec_i[i_tile + ii],     \
                                                 vec_j[j_tile + jj], offset, depth_simd_dimensions);                   \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (remainder_depth > 0) {                                                                                     \
            for (nk_size_t i = 0; i < macro_i_size; i++) {                                                             \
                partial_load_vec_fn(vector_base_ptrs_i[i] + aligned_depth, &vec_i[i], remainder_dimensions);           \
            }                                                                                                          \
            for (nk_size_t j = 0; j < macro_j_size; j++) {                                                             \
                partial_load_vec_fn(vector_base_ptrs_j[j] + aligned_depth, &vec_j[j], remainder_dimensions);           \
            }                                                                                                          \
            nk_size_t offset = aligned_depth * dimensions_per_value;                                                   \
            for (nk_size_t i_tile = 0; i_tile < macro_i_size; i_tile += 4) {                                           \
                for (nk_size_t j_tile = 0; j_tile < macro_j_size; j_tile += 4) {                                       \
                    nk_size_t tile_i_size = (i_tile + 4 <= macro_i_size) ? 4 : (macro_i_size - i_tile);                \
                    nk_size_t tile_j_size = (j_tile + 4 <= macro_j_size) ? 4 : (macro_j_size - j_tile);                \
                    for (nk_size_t ii = 0; ii < tile_i_size; ii++) {                                                   \
                        for (nk_size_t jj = 0; jj < tile_j_size; jj++) {                                               \
                            inner_product_fn(&accumulator_tiles[i_tile + ii][j_tile + jj], vec_i[i_tile + ii],         \
                                             vec_j[j_tile + jj], offset, remainder_dimensions);                        \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        /* Finalize and store (no mirroring - off-diagonal already in upper triangle) */                               \
        for (nk_size_t ii = 0; ii < macro_i_size; ii++) {                                                              \
            nk_size_t jj = 0;                                                                                          \
            /* Process 4 at a time */                                                                                  \
            for (; jj + 4 <= macro_j_size; jj += 4) {                                                                  \
                result_vec_type result_vec;                                                                            \
                reduce_accumulators_fn(&accumulator_tiles[ii][jj], &accumulator_tiles[ii][jj + 1],                     \
                                       &accumulator_tiles[ii][jj + 2], &accumulator_tiles[ii][jj + 3], depth,          \
                                       &result_vec);                                                                   \
                nk_##result_value_type##_t *out_ptr = &result[(i_macro + ii) * result_stride_values + (j_macro + jj)]; \
                partial_store_fn(&result_vec, out_ptr, 4);                                                             \
            }                                                                                                          \
            /* Handle remaining values (< 4) */                                                                        \
            if (jj < macro_j_size) {                                                                                   \
                nk_size_t remaining = macro_j_size - jj;                                                               \
                result_vec_type result_vec;                                                                            \
                reduce_accumulators_fn(&accumulator_tiles[ii][jj], &accumulator_tiles[ii][jj + 1],                     \
                                       &accumulator_tiles[ii][jj + 2], &accumulator_tiles[ii][jj + 3], depth,          \
                                       &result_vec);                                                                   \
                nk_##result_value_type##_t *out_ptr = &result[(i_macro + ii) * result_stride_values + (j_macro + jj)]; \
                partial_store_fn(&result_vec, out_ptr, remaining);                                                     \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                         \
        nk_##input_value_type##_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,              \
        nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {       \
        nk_size_t const macro_tile_size = 32;                                                                          \
        nk_size_t const finalizer_batch_size = 4;                                                                      \
        nk_size_t const row_block_size = 128;     /* L2 cache blocking */                                              \
        nk_size_t const column_block_size = 2048; /* L3 cache blocking */                                              \
                                                                                                                       \
        /* Stride and depth calculations */                                                                            \
        nk_size_t const vectors_stride_values = stride / sizeof(nk_##input_value_type##_t);                            \
        nk_size_t const result_stride_values = result_stride / sizeof(nk_##result_value_type##_t);                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;           \
                                                                                                                       \
        /* Process upper triangle with L3/L2/L1 blocking (column blocks → row blocks → 32×32 macro-tiles) */           \
        for (nk_size_t j_block = 0; j_block < n_vectors; j_block += column_block_size) {                               \
            nk_size_t j_block_end = (j_block + column_block_size < n_vectors) ? j_block + column_block_size            \
                                                                              : n_vectors;                             \
                                                                                                                       \
            for (nk_size_t i_block = row_start; i_block < row_end; i_block += row_block_size) {                        \
                nk_size_t i_block_end = (i_block + row_block_size < row_end) ? i_block + row_block_size : row_end;     \
                                                                                                                       \
                /* Skip blocks entirely below diagonal (i_block_end <= j_block) */                                     \
                if (i_block_end <= j_block) continue;                                                                  \
                                                                                                                       \
                for (nk_size_t i_macro = i_block; i_macro < i_block_end; i_macro += macro_tile_size) {                 \
                    /* Upper triangle: j_macro starts at max(i_macro, j_block) */                                      \
                    nk_size_t j_start = (i_macro > j_block) ? i_macro : j_block;                                       \
                    for (nk_size_t j_macro = j_start; j_macro < j_block_end; j_macro += macro_tile_size) {             \
                        nk_size_t macro_i_size = (i_macro + macro_tile_size <= i_block_end) ? macro_tile_size          \
                                                                                            : (i_block_end - i_macro); \
                        nk_size_t macro_j_size = (j_macro + macro_tile_size <= j_block_end) ? macro_tile_size          \
                                                                                            : (j_block_end - j_macro); \
                        state_type accumulator_tiles[32][32];                                                          \
                                                                                                                       \
                        /* Hoist pointer computation outside depth loop */                                             \
                        nk_##input_value_type##_t const *vector_base_ptrs_i[32];                                       \
                        nk_##input_value_type##_t const *vector_base_ptrs_j[32];                                       \
                        for (nk_size_t i = 0; i < macro_i_size; i++) {                                                 \
                            vector_base_ptrs_i[i] = vectors + (i_macro + i) * vectors_stride_values;                   \
                        }                                                                                              \
                        if (i_macro != j_macro) { /* Off-diagonal needs both sets */                                   \
                            for (nk_size_t j = 0; j < macro_j_size; j++) {                                             \
                                vector_base_ptrs_j[j] = vectors + (j_macro + j) * vectors_stride_values;               \
                            }                                                                                          \
                        }                                                                                              \
                                                                                                                       \
                        if (i_macro == j_macro) {                                                                      \
                            /* Diagonal macro-tile: load vec[32] once (50% load reduction) */                          \
                            nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                    \
                                accumulator_tiles, vector_base_ptrs_i, i_macro, macro_i_size, aligned_depth,           \
                                remainder_depth, remainder_dimensions, depth_step_values, dimensions_per_value,        \
                                result, result_stride_values, finalizer_batch_size, depth);                            \
                        }                                                                                              \
                        else {                                                                                         \
                            /* Off-diagonal macro-tile: load vec_i[32] + vec_j[32] */                                  \
                            nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal##_(               \
                                accumulator_tiles, vector_base_ptrs_i, vector_base_ptrs_j, i_macro, j_macro,           \
                                macro_i_size, macro_j_size, aligned_depth, remainder_depth, remainder_dimensions,      \
                                depth_step_values, dimensions_per_value, result, result_stride_values,                 \
                                finalizer_batch_size, depth);                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }
/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes) */
nk_define_cross_pack_size_(dots, f64, serial, f64, f64, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, serial, f64, f64, nk_assign_from_to_, /*depth_simd_dimensions=*/2,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, serial, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                           nk_dot_f64x2_update_serial, nk_dot_f64x2_finalize_serial, nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, serial, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                        nk_load_b128_serial_, nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_serial,
                        nk_dot_f64x2_finalize_serial, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes) */
nk_define_cross_pack_size_(dots, f32, serial, f32, f32, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, serial, f32, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, serial, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                           nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                           nk_dot_f32x4_update_serial, nk_dot_f32x4_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, serial, f32, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                        nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_serial,
                        nk_dot_f32x4_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, f16, serial, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, serial, f16, f16, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, serial, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_serial, nk_dot_f16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, serial, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_serial,
                        nk_dot_f16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, bf16, serial, bf16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, serial, bf16, bf16, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, serial, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, serial, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_serial,
                        nk_dot_bf16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes), I32 accumulator */
nk_define_cross_pack_size_(dots, i8, serial, i8, i8, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, serial, i8, i8, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, serial, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_serial, nk_dot_i8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, serial, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_serial,
                        nk_dot_i8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes), U32 accumulator */
nk_define_cross_pack_size_(dots, u8, serial, u8, u8, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, serial, u8, u8, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, serial, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, serial, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_serial,
                        nk_dot_u8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=16 (16 e4m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e4m3, serial, e4m3, e4m3, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, serial, e4m3, e4m3, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, serial, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e4m3x16_update_serial, nk_dot_e4m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, serial, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_serial,
                        nk_dot_e4m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=16 (16 e5m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e5m2, serial, e5m2, e5m2, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, serial, e5m2, e5m2, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, serial, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e5m2x16_update_serial, nk_dot_e5m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, serial, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_serial,
                        nk_dot_e5m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=16 (16 e2m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e2m3, serial, e2m3, e2m3, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, serial, e2m3, e2m3, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, serial, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e2m3x16_update_serial, nk_dot_e2m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, serial, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_serial,
                        nk_dot_e2m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=16 (16 e3m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e3m2, serial, e3m2, e3m2, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, serial, e3m2, e3m2, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, serial, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e3m2x16_update_serial, nk_dot_e3m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, serial, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_serial,
                        nk_dot_e3m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U4 GEMM: u4x2 for both A and B */
nk_define_cross_pack_size_(dots, u4, serial, u4x2, u4x2, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, u4, serial, u4x2, u4x2, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, u4, serial, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_u4x16_update_serial, nk_dot_u4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, serial, u4x2, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_u4x16_update_serial,
                        nk_dot_u4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* I4 GEMM: i4x2 for both A and B */
nk_define_cross_pack_size_(dots, i4, serial, i4x2, i4x2, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, i4, serial, i4x2, i4x2, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, i4, serial, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, i4, serial, i4x2, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_i4x16_update_serial,
                        nk_dot_i4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride_in_bytes, writes BF16 tightly packed (stride = column_count × sizeof(bf16)).
 */
NK_PUBLIC void nk_dots_compact_bf16_serial(void *c, nk_size_t row_count, nk_size_t column_count,
                                           nk_size_t c_stride_in_bytes) {
    nk_size_t const c_stride_in_values = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_f32_t const *source_row = c_f32 + row_index * c_stride_in_values;
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
    nk_size_t const c_stride_in_values = c_stride_in_bytes / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_i32_t const *source_row = c_i32 + row_index * c_stride_in_values;
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
