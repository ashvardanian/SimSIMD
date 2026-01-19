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
#include "numkong/binary/serial.h" // `nk_popcount_u1`
#include "numkong/cast/serial.h"   // Generic load/store helpers
#include "numkong/dot/serial.h"    // Stateful dot product helpers

#if defined(__cplusplus)
extern "C" {
#endif

/*  Packed buffer header (64-byte aligned).
 *  Used by all packed matmul backends (serial, NEON, AVX-512, SVE).
 */
typedef struct {
    nk_u32_t column_groups_count; // Number of column groups (ceil(column_count / group_size))
    nk_u32_t depth;               // depth dimension (contiguous within each row)
    nk_u16_t group_size;          // Columns per group (16)
    nk_u16_t column_remainder;    // Remaining columns in last group
    nk_u32_t reserved[12];        // Padding to 64 bytes
} nk_dots_packed_buffer_header_t;

#define NK_DOTS_GROUP_SIZE_ 16 // Rows per group for alignment

/** @brief Regular fused-multiply-add for standard numeric types (f32, i8, u8, etc.) */
#define nk_fma_multiply_add_(acc, a, b) (*(acc) += (a) * (b))

/** @brief Fused-multiply-add for bit-vectors accumulating intersection population counts, like Jaccard kernels. */
#define nk_fma_u1_and_popcnt_(acc, a, b) (*(acc) += nk_popcount_u1((nk_u1x8_t)((a) & (b))))

/** @brief Fused-multily-add for 4-bit unsigned nibbles. */
#define nk_fma_u4_nibble_dot_(acc, a, b) \
    ((*(acc) += ((a) & 0x0F) * ((b) & 0x0F)), (*(acc) += (((a) >> 4) & 0x0F) * (((b) >> 4) & 0x0F)))

/** @brief Fused-multily-add for 4-bit signed nibbles. */
#define nk_fma_i4_nibble_dot_(acc, a, b)                                                   \
    (*(acc) += (((nk_i32_t)(((a) & 0x0F) ^ 8) - 8) * ((nk_i32_t)(((b) & 0x0F) ^ 8) - 8)) + \
               (((nk_i32_t)((((a) >> 4) & 0x0F) ^ 8) - 8) * ((nk_i32_t)((((b) >> 4) & 0x0F) ^ 8) - 8)))

/**
 *  @brief Macro to generate packed_size function for transpose-based packing.
 *
 *  Calculates buffer size needed for packed B matrix: header + column_groups_count × group_size × depth.
 *  Edge rows are zero-padded to full group size for uniform SIMD loads.
 */
#define nk_define_dots_pack_size_(suffix, input_type, storage_type, output_type)                               \
    NK_PUBLIC nk_size_t nk_dots_packed_size_##input_type##_##suffix(nk_size_t column_count, nk_size_t depth) { \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;                                                      \
        nk_size_t const column_groups_count = (column_count + group_size - 1) / group_size;                    \
        return sizeof(nk_dots_packed_buffer_header_t) +                                                        \
               column_groups_count * group_size * depth * sizeof(nk_##storage_type##_t);                       \
    }

/**
 *  @brief Macro to generate pack function for grouped row-major packing.
 *
 *  Packs B[column_count, depth] into B_packed[column_count_padded, depth] with row grouping.
 *  B is already in "transposed" form (column_count rows, depth columns) as per the API convention.
 *  Each group contains group_size rows, each row has depth contiguous elements.
 *  Edge groups are zero-padded to full group_size.
 */
#define nk_define_dots_pack_(suffix, input_type, storage_type, output_type, scalar_convert_fn)                       \
    NK_PUBLIC void nk_dots_pack_##input_type##_##suffix(nk_##input_type##_t const *b, nk_size_t column_count,        \
                                                        nk_size_t depth, nk_size_t b_stride_in_bytes,                \
                                                        void *b_packed) {                                            \
                                                                                                                     \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;                                                            \
        nk_size_t const column_groups_count = (column_count + group_size - 1) / group_size;                          \
                                                                                                                     \
        /* Store dimensions in header */                                                                             \
        nk_dots_packed_buffer_header_t *header = (nk_dots_packed_buffer_header_t *)b_packed;                         \
        header->column_groups_count = (nk_u32_t)column_groups_count;                                                 \
        header->depth = (nk_u32_t)depth;                                                                             \
        header->group_size = (nk_u16_t)group_size;                                                                   \
        header->column_remainder = (nk_u16_t)(column_count % group_size);                                            \
                                                                                                                     \
        nk_##storage_type##_t *packed = (nk_##storage_type##_t *)((char *)b_packed +                                 \
                                                                  sizeof(nk_dots_packed_buffer_header_t));           \
                                                                                                                     \
        /* Zero entire buffer for edge padding */                                                                    \
        nk_size_t const total_elements = column_groups_count * group_size * depth;                                   \
        for (nk_size_t i = 0; i < total_elements; ++i) packed[i] = 0;                                                \
                                                                                                                     \
        /* Copy/convert B[column_count, depth] to packed[column_count_padded, depth] with row grouping */            \
        /* B[j, depth_index] at address B + j * b_stride_in_bytes + depth_index * sizeof(element) */                 \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                              \
            nk_size_t const group_index = column_index / group_size;                                                 \
            nk_size_t const column_index_in_group = column_index % group_size;                                       \
            nk_##storage_type##_t *destination_row = packed + group_index * group_size * depth +                     \
                                                     column_index_in_group * depth;                                  \
            nk_##input_type##_t const *source_row = (nk_##input_type##_t const *)((char const *)b +                  \
                                                                                  column_index * b_stride_in_bytes); \
                                                                                                                     \
            for (nk_size_t depth_index = 0; depth_index < depth; ++depth_index) {                                    \
                scalar_convert_fn(&source_row[depth_index], &destination_row[depth_index]);                          \
            }                                                                                                        \
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
 *      for (depth_index = 0; depth_index < aligned_depth; depth_index += simd_width) {
 *          load_fn(b_depth_ptr_0 + depth_index, &b_vector_0);
 *          load_fn(b_depth_ptr_1 + depth_index, &b_vector_1);
 *          // ... 16 FMAs for 4 × 4 register tile
 *      }
 */
#define nk_define_dots_packed_4x4_vectors_aligned_(                                                                    \
    suffix, input_type, storage_type, output_type, vec_type, state_type, result_vec_type, init_fn, load_a_fn,          \
    partial_load_a_fn, load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, simd_width)             \
                                                                                                                       \
    NK_PUBLIC void nk_dots_##suffix##_aligned_(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,       \
                                               nk_##output_type##_t *c_matrix, nk_size_t row_count,                    \
                                               nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,   \
                                               nk_size_t c_stride_in_bytes) {                                          \
        nk_##storage_type##_t const *packed_data =                                                                     \
            (nk_##storage_type##_t const *)((char const *)b_packed_buffer + sizeof(nk_dots_packed_buffer_header_t));   \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;              /* L2 cache blocking over rows */                           \
        nk_size_t const column_block_size = 2048;          /* L3 cache blocking over columns */                        \
        nk_size_t const register_row_count = 4;            /* Rows per register tile */                                \
        nk_size_t const register_column_count = 4;         /* Columns per register tile */                             \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;  /* Columns per packed group */                              \
        nk_size_t const group_stride = group_size * depth; /* Elements per group */                                    \
        nk_size_t const aligned_depth = (depth / simd_width) * simd_width;                                             \
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
                    /* Compute B pointers once per column tile (outside row and depth loops) */                        \
                    nk_size_t const group_index = tile_column_start_index / group_size;                                \
                    nk_size_t const column_index_in_group = tile_column_start_index % group_size;                      \
                    nk_##storage_type##_t const *group_base_ptr = packed_data + group_index * group_stride;            \
                                                                                                                       \
                    nk_##storage_type##_t const *b_depth_ptr_0 = group_base_ptr + (column_index_in_group + 0) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_1 = group_base_ptr + (column_index_in_group + 1) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_2 = group_base_ptr + (column_index_in_group + 2) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_3 = group_base_ptr + (column_index_in_group + 3) * depth; \
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
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth; depth_index += simd_width) {      \
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
                            update_fn(&accumulator_tiles[0][0], a_vector_0, b_vector_0);                               \
                            update_fn(&accumulator_tiles[0][1], a_vector_0, b_vector_1);                               \
                            update_fn(&accumulator_tiles[0][2], a_vector_0, b_vector_2);                               \
                            update_fn(&accumulator_tiles[0][3], a_vector_0, b_vector_3);                               \
                            update_fn(&accumulator_tiles[1][0], a_vector_1, b_vector_0);                               \
                            update_fn(&accumulator_tiles[1][1], a_vector_1, b_vector_1);                               \
                            update_fn(&accumulator_tiles[1][2], a_vector_1, b_vector_2);                               \
                            update_fn(&accumulator_tiles[1][3], a_vector_1, b_vector_3);                               \
                            update_fn(&accumulator_tiles[2][0], a_vector_2, b_vector_0);                               \
                            update_fn(&accumulator_tiles[2][1], a_vector_2, b_vector_1);                               \
                            update_fn(&accumulator_tiles[2][2], a_vector_2, b_vector_2);                               \
                            update_fn(&accumulator_tiles[2][3], a_vector_2, b_vector_3);                               \
                            update_fn(&accumulator_tiles[3][0], a_vector_3, b_vector_0);                               \
                            update_fn(&accumulator_tiles[3][1], a_vector_3, b_vector_1);                               \
                            update_fn(&accumulator_tiles[3][2], a_vector_3, b_vector_2);                               \
                            update_fn(&accumulator_tiles[3][3], a_vector_3, b_vector_3);                               \
                        }                                                                                              \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        result_vec_type result_vector;                                                                 \
                        nk_##output_type##_t *c_row_ptr_0 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 0) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[0][0], &accumulator_tiles[0][1], &accumulator_tiles[0][2],      \
                                    &accumulator_tiles[0][3], &result_vector);                                         \
                        partial_store_fn(&result_vector, c_row_ptr_0 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_1 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 1) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[1][0], &accumulator_tiles[1][1], &accumulator_tiles[1][2],      \
                                    &accumulator_tiles[1][3], &result_vector);                                         \
                        partial_store_fn(&result_vector, c_row_ptr_1 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_2 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 2) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[2][0], &accumulator_tiles[2][1], &accumulator_tiles[2][2],      \
                                    &accumulator_tiles[2][3], &result_vector);                                         \
                        partial_store_fn(&result_vector, c_row_ptr_2 + tile_column_start_index, 4);                    \
                        nk_##output_type##_t *c_row_ptr_3 = (nk_##output_type##_t *)((char *)c_matrix +                \
                                                                                     (tile_row_start_index + 3) *      \
                                                                                         c_stride_in_bytes);           \
                        finalize_fn(&accumulator_tiles[3][0], &accumulator_tiles[3][1], &accumulator_tiles[3][2],      \
                                    &accumulator_tiles[3][3], &result_vector);                                         \
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
#define nk_define_dots_packed_1x8_vectors_aligned_(                                                                    \
    suffix, input_type, storage_type, output_type, vec_type, state_type, result_vec_type, init_fn, load_a_fn,          \
    partial_load_a_fn, load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, simd_width)             \
                                                                                                                       \
    NK_PUBLIC void nk_dots_##suffix##_1x8_aligned_(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,   \
                                                   nk_##output_type##_t *c_matrix, nk_size_t row_count,                \
                                                   nk_size_t column_count, nk_size_t depth,                            \
                                                   nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {         \
        nk_##storage_type##_t const *packed_data =                                                                     \
            (nk_##storage_type##_t const *)((char const *)b_packed_buffer + sizeof(nk_dots_packed_buffer_header_t));   \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;              /* L2 cache blocking over rows */                           \
        nk_size_t const column_block_size = 2048;          /* L3 cache blocking over columns */                        \
        nk_size_t const register_row_count = 1;            /* Rows per register tile */                                \
        nk_size_t const register_column_count = 8;         /* Columns per register tile (2 × 4) */                     \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;  /* Columns per packed group */                              \
        nk_size_t const group_stride = group_size * depth; /* Elements per group */                                    \
        nk_size_t const aligned_depth = (depth / simd_width) * simd_width;                                             \
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
                    /* Compute B pointers once per column tile (outside row and depth loops) */                        \
                    nk_size_t const group_index = tile_column_start_index / group_size;                                \
                    nk_size_t const column_index_in_group = tile_column_start_index % group_size;                      \
                    nk_##storage_type##_t const *group_base_ptr = packed_data + group_index * group_stride;            \
                                                                                                                       \
                    nk_##storage_type##_t const *b_depth_ptr_0 = group_base_ptr + (column_index_in_group + 0) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_1 = group_base_ptr + (column_index_in_group + 1) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_2 = group_base_ptr + (column_index_in_group + 2) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_3 = group_base_ptr + (column_index_in_group + 3) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_4 = group_base_ptr + (column_index_in_group + 4) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_5 = group_base_ptr + (column_index_in_group + 5) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_6 = group_base_ptr + (column_index_in_group + 6) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_7 = group_base_ptr + (column_index_in_group + 7) * depth; \
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
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth; depth_index += simd_width) {      \
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
                            update_fn(&accumulator_0, a_vector, b_vector_0);                                           \
                            update_fn(&accumulator_1, a_vector, b_vector_1);                                           \
                            update_fn(&accumulator_2, a_vector, b_vector_2);                                           \
                            update_fn(&accumulator_3, a_vector, b_vector_3);                                           \
                            update_fn(&accumulator_4, a_vector, b_vector_4);                                           \
                            update_fn(&accumulator_5, a_vector, b_vector_5);                                           \
                            update_fn(&accumulator_6, a_vector, b_vector_6);                                           \
                            update_fn(&accumulator_7, a_vector, b_vector_7);                                           \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store 1 × 8 results using two 4-way reductions */                              \
                        result_vec_type result_vector;                                                                 \
                        nk_##output_type##_t *c_row_ptr = (nk_##output_type##_t *)((char *)c_matrix +                  \
                                                                                   row_index * c_stride_in_bytes);     \
                        /* First 4 columns */                                                                          \
                        finalize_fn(&accumulator_0, &accumulator_1, &accumulator_2, &accumulator_3, &result_vector);   \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index, 4);                      \
                        /* Second 4 columns */                                                                         \
                        finalize_fn(&accumulator_4, &accumulator_5, &accumulator_6, &accumulator_7, &result_vector);   \
                        partial_store_fn(&result_vector, c_row_ptr + tile_column_start_index + 4, 4);                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/* Generate both aligned kernel variants for each platform */
#define nk_define_dots_packed_(suffix, input_type, storage_type, output_type, vec_type, state_type, result_vec_type,   \
                               init_fn, load_a_fn, partial_load_a_fn, load_b_fn, partial_load_b_fn, update_fn,         \
                               finalize_fn, partial_store_fn, simd_width)                                              \
    /* Generate 4 × 4 aligned kernel */                                                                                \
    nk_define_dots_packed_4x4_vectors_aligned_(                                                                        \
        suffix, input_type, storage_type, output_type, vec_type, state_type, result_vec_type, init_fn, load_a_fn,      \
        partial_load_a_fn, load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, simd_width)         \
    /* Generate 1 × 8 aligned kernel */                                                                                \
    nk_define_dots_packed_1x8_vectors_aligned_(                                                                        \
        suffix, input_type, storage_type, output_type, vec_type, state_type, result_vec_type, init_fn, load_a_fn,      \
        partial_load_a_fn, load_b_fn, partial_load_b_fn, update_fn, finalize_fn, partial_store_fn, simd_width)         \
                                                                                                                       \
    NK_PUBLIC void nk_dots_packed_##suffix(nk_##input_type##_t const *a_matrix, void const *b_packed_buffer,           \
                                           nk_##output_type##_t *c_matrix, nk_size_t row_count,                        \
                                           nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,       \
                                           nk_size_t c_stride_in_bytes) {                                              \
        /* Cache blocking parameters (hardcoded for optimal L1/L2/L3 utilization) */                                   \
        nk_size_t const row_block_size = 128;              /* L2 cache blocking over rows */                           \
        nk_size_t const column_block_size = 2048;          /* L3 cache blocking over columns */                        \
        nk_size_t const register_row_count = 4;            /* Rows per register tile */                                \
        nk_size_t const register_column_count = 4;         /* Columns per register tile */                             \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;  /* Columns per packed group */                              \
        nk_size_t const group_stride = group_size * depth; /* Elements per group */                                    \
        (void)register_column_count;                                                                                   \
        (void)group_stride; /* Suppress unused warnings */                                                             \
        /* Use 1 × 8 kernel when columns are aligned to 8 and many columns relative to rows */                         \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % simd_width == 0) {                       \
            nk_dots_##suffix##_1x8_aligned_(a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth,       \
                                            a_stride_in_bytes, c_stride_in_bytes);                                     \
            return;                                                                                                    \
        }                                                                                                              \
        /* Use 4 × 4 kernel when dimensions are 4-aligned */                                                           \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % simd_width == 0) {                                  \
            nk_dots_##suffix##_aligned_(a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth,           \
                                        a_stride_in_bytes, c_stride_in_bytes);                                         \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Compute aligned/remainder depth for partial loads */                                                        \
        nk_size_t const aligned_depth = (depth / simd_width) * simd_width;                                             \
        nk_size_t const remainder_depth = depth - aligned_depth;                                                       \
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
                    /* Compute B pointers once per column tile (outside row and depth loops) */                        \
                    nk_size_t const group_index = tile_column_start_index / group_size;                                \
                    nk_size_t const column_index_in_group = tile_column_start_index % group_size;                      \
                    nk_##storage_type##_t const *group_base_ptr = packed_data + group_index * group_stride;            \
                                                                                                                       \
                    nk_##storage_type##_t const *b_depth_ptr_0 = group_base_ptr + (column_index_in_group + 0) * depth; \
                    nk_##storage_type##_t const *b_depth_ptr_1 = (tile_column_count > 1)                               \
                                                                     ? group_base_ptr +                                \
                                                                           (column_index_in_group + 1) * depth         \
                                                                     : b_depth_ptr_0;                                  \
                    nk_##storage_type##_t const *b_depth_ptr_2 = (tile_column_count > 2)                               \
                                                                     ? group_base_ptr +                                \
                                                                           (column_index_in_group + 2) * depth         \
                                                                     : b_depth_ptr_0;                                  \
                    nk_##storage_type##_t const *b_depth_ptr_3 = (tile_column_count > 3)                               \
                                                                     ? group_base_ptr +                                \
                                                                           (column_index_in_group + 3) * depth         \
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
                        for (nk_size_t k = 0; k < aligned_depth; k += simd_width) {                                    \
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
                            update_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec);                             \
                            update_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec);                            \
                            update_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec);                             \
                            update_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec);                            \
                            update_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec);                            \
                            update_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec);                           \
                            update_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec);                            \
                            update_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec);                           \
                            update_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec);                             \
                            update_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec);                            \
                            update_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec);                             \
                            update_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec);                            \
                            update_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec);                            \
                            update_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec);                           \
                            update_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec);                            \
                            update_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec);                           \
                        }                                                                                              \
                                                                                                                       \
                        /* Handle remainder k positions with partial loads */                                          \
                        if (remainder_depth > 0) {                                                                     \
                            /* Load next few elements from 4 rows from A */                                            \
                            partial_load_a_fn(a_row_ptr_0 + aligned_depth, &a_first_vec, remainder_depth);             \
                            partial_load_a_fn(a_row_ptr_1 + aligned_depth, &a_second_vec, remainder_depth);            \
                            partial_load_a_fn(a_row_ptr_2 + aligned_depth, &a_third_vec, remainder_depth);             \
                            partial_load_a_fn(a_row_ptr_3 + aligned_depth, &a_fourth_vec, remainder_depth);            \
                                                                                                                       \
                            /* Load next few elements from 4 rows from B */                                            \
                            partial_load_b_fn(b_depth_ptr_0 + aligned_depth, &b_first_vec, remainder_depth);           \
                            partial_load_b_fn(b_depth_ptr_1 + aligned_depth, &b_second_vec, remainder_depth);          \
                            partial_load_b_fn(b_depth_ptr_2 + aligned_depth, &b_third_vec, remainder_depth);           \
                            partial_load_b_fn(b_depth_ptr_3 + aligned_depth, &b_fourth_vec, remainder_depth);          \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            update_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec);                             \
                            update_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec);                            \
                            update_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec);                             \
                            update_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec);                            \
                            update_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec);                            \
                            update_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec);                           \
                            update_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec);                            \
                            update_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec);                           \
                            update_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec);                             \
                            update_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec);                            \
                            update_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec);                             \
                            update_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec);                            \
                            update_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec);                            \
                            update_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec);                           \
                            update_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec);                            \
                            update_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec);                           \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            result_vec_type result_vector;                                                             \
                            finalize_fn(&accumulator_tiles[r][0], &accumulator_tiles[r][1], &accumulator_tiles[r][2],  \
                                        &accumulator_tiles[r][3], &result_vector);                                     \
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
 * - Vector loads (simd_width elements per load)
 * - Register reuse (row_i loaded once per iteration, used for all j ≥ i)
 * - State-based accumulation (supports Neumaier compensation, platform-specific precision)
 */
#define nk_define_dots_symmetric_(suffix, input_type, output_type, vec_type, state_type, result_vec_type, init_fn, \
                                  load_fn, partial_load_fn, update_fn, finalize_fn, simd_width)                    \
    NK_PUBLIC void nk_dots_symmetric_##suffix(nk_##input_type##_t const *vectors, nk_size_t n_vectors,             \
                                              nk_size_t depth, nk_size_t stride, nk_##output_type##_t *result,     \
                                              nk_size_t result_stride) {                                           \
                                                                                                                   \
        nk_size_t const vectors_stride_elements = stride / sizeof(nk_##input_type##_t);                            \
        nk_size_t const result_stride_elements = result_stride / sizeof(nk_##output_type##_t);                     \
        nk_size_t const aligned_depth = (depth / simd_width) * simd_width;                                         \
        nk_size_t const remainder_depth = depth - aligned_depth;                                                   \
                                                                                                                   \
        /* Compute upper triangle including diagonal */                                                            \
        for (nk_size_t i = 0; i < n_vectors; i++) {                                                                \
            nk_##input_type##_t const *row_i = vectors + i * vectors_stride_elements;                              \
            for (nk_size_t j = i; j < n_vectors; j++) {                                                            \
                nk_##input_type##_t const *row_j = vectors + j * vectors_stride_elements;                          \
                                                                                                                   \
                /* Initialize accumulator state */                                                                 \
                state_type acc;                                                                                    \
                init_fn(&acc);                                                                                     \
                                                                                                                   \
                /* Vectorized depth loop */                                                                        \
                for (nk_size_t d = 0; d < aligned_depth; d += simd_width) {                                        \
                    vec_type vec_i, vec_j;                                                                         \
                    load_fn(row_i + d, &vec_i);                                                                    \
                    load_fn(row_j + d, &vec_j);                                                                    \
                    update_fn(&acc, vec_i, vec_j);                                                                 \
                }                                                                                                  \
                                                                                                                   \
                /* Handle remainder with partial load */                                                           \
                if (remainder_depth > 0) {                                                                         \
                    vec_type vec_i, vec_j;                                                                         \
                    partial_load_fn(row_i + aligned_depth, &vec_i, remainder_depth);                               \
                    partial_load_fn(row_j + aligned_depth, &vec_j, remainder_depth);                               \
                    update_fn(&acc, vec_i, vec_j);                                                                 \
                }                                                                                                  \
                                                                                                                   \
                /* Finalize: horizontal reduction to scalar */                                                     \
                state_type dummy_b, dummy_c, dummy_d;                                                              \
                init_fn(&dummy_b);                                                                                 \
                init_fn(&dummy_c);                                                                                 \
                init_fn(&dummy_d);                                                                                 \
                result_vec_type result_vec;                                                                        \
                finalize_fn(&acc, &dummy_b, &dummy_c, &dummy_d, &result_vec);                                      \
                                                                                                                   \
                /* Store result and mirror to lower triangle */                                                    \
                nk_##output_type##_t val = result_vec.output_type##s[0];                                           \
                result[i * result_stride_elements + j] = val;                                                      \
                if (i != j) { result[j * result_stride_elements + i] = val; }                                      \
            }                                                                                                      \
        }                                                                                                          \
    }

/* F64 GEMM: simd_width=2 (2 f64s = 16 bytes) */
nk_define_dots_pack_size_(serial, f64, f64, f64)
nk_define_dots_pack_(serial, f64, f64, f64, nk_assign_from_to_)
nk_define_dots_symmetric_(f64_serial, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                          nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                          nk_dot_f64x2_update_serial, nk_dot_f64x2_finalize_serial, /*simd_width=*/2)
nk_define_dots_packed_(f64_serial, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                       nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                       nk_load_b128_serial_, nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_serial,
                       nk_dot_f64x2_finalize_serial, nk_partial_store_b64x4_serial_,
                       /*simd_width=*/2)

/* F32 GEMM: simd_width=4 (4 f32s = 16 bytes) */
nk_define_dots_pack_size_(serial, f32, f32, f32)
nk_define_dots_pack_(serial, f32, f32, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(f32_serial, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                          nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                          nk_dot_f32x4_update_serial, nk_dot_f32x4_finalize_serial, /*simd_width=*/4)
nk_define_dots_packed_(f32_serial, f32, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b128_vec_t,
                       nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                       nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_serial,
                       nk_dot_f32x4_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/4)

/* F16 GEMM: simd_width=8 (8 f16s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(serial, f16, f32, f32)
nk_define_dots_pack_(serial, f16, f16, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(f16_serial, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                          nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                          nk_dot_f16x8_update_serial, nk_dot_f16x8_finalize_serial, /*simd_width=*/8)
nk_define_dots_packed_(f16_serial, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                       nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                       nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_serial,
                       nk_dot_f16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* BF16 GEMM: simd_width=8 (8 bf16s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(serial, bf16, f32, f32)
nk_define_dots_pack_(serial, bf16, bf16, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(bf16_serial, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                          nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                          nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial, /*simd_width=*/8)
nk_define_dots_packed_(bf16_serial, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                       nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                       nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_serial,
                       nk_dot_bf16x8_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* I8 GEMM: simd_width=16 (16 i8s = 16 bytes), I32 accumulator */
nk_define_dots_pack_size_(serial, i8, i8, i32)
nk_define_dots_pack_(serial, i8, i8, i32, nk_assign_from_to_)
nk_define_dots_symmetric_(i8_serial, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_i8x16_update_serial, nk_dot_i8x16_finalize_serial, /*simd_width=*/16)
nk_define_dots_packed_(i8_serial, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_serial,
                       nk_dot_i8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* U8 GEMM: simd_width=16 (16 u8s = 16 bytes), U32 accumulator */
nk_define_dots_pack_size_(serial, u8, u8, u32)
nk_define_dots_pack_(serial, u8, u8, u32, nk_assign_from_to_)
nk_define_dots_symmetric_(u8_serial, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial, /*simd_width=*/16)
nk_define_dots_packed_(u8_serial, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_serial,
                       nk_dot_u8x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* E4M3 GEMM: simd_width=16 (16 e4m3s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(serial, e4m3, e4m3, f32)
nk_define_dots_pack_(serial, e4m3, e4m3, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(e4m3_serial, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e4m3x16_update_serial, nk_dot_e4m3x16_finalize_serial, /*simd_width=*/16)
nk_define_dots_packed_(e4m3_serial, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_serial,
                       nk_dot_e4m3x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* E5M2 GEMM: simd_width=16 (16 e5m2s = 16 bytes), F32 accumulator */
nk_define_dots_pack_size_(serial, e5m2, e5m2, f32)
nk_define_dots_pack_(serial, e5m2, e5m2, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(e5m2_serial, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                          nk_dot_e5m2x16_update_serial, nk_dot_e5m2x16_finalize_serial, /*simd_width=*/16)
nk_define_dots_packed_(e5m2_serial, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_serial,
                       nk_dot_e5m2x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* U4 GEMM (SYMMETRIC - temporarily reverting asymmetric attempt): u4x2 for both A and B */
/* TODO: Asymmetric u4x2→u8 expansion requires custom pack logic beyond macro capabilities */
nk_define_dots_pack_size_(serial, u4x2, u4x2, u32)
nk_define_dots_pack_(serial, u4x2, u4x2, u32, nk_assign_from_to_)
nk_define_dots_symmetric_(u4_serial, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                          nk_dot_u4x16_update_serial, nk_dot_u4x16_finalize_serial, /*simd_width=*/8)
nk_define_dots_packed_(u4_serial, u4x2, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                       nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_u4x16_update_serial,
                       nk_dot_u4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* I4 GEMM (SYMMETRIC - temporarily reverting asymmetric attempt): i4x2 for both A and B */
/* TODO: Asymmetric i4x2→i8 expansion requires custom pack logic beyond macro capabilities */
nk_define_dots_pack_size_(serial, i4x2, i4x2, i32)
nk_define_dots_pack_(serial, i4x2, i4x2, i32, nk_assign_from_to_)
nk_define_dots_symmetric_(i4_serial, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                          nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                          nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial, /*simd_width=*/8)
nk_define_dots_packed_(i4_serial, i4x2, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                       nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                       nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_i4x16_update_serial,
                       nk_dot_i4x16_finalize_serial, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

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
