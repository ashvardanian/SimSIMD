/**
 *  @brief Portable GEMM macros and serial implementations for matrix multiplication.
 *  @file include/numkong/dots/serial.h
 *  @sa include/numkong/dots.h for API overview and use cases
 *  @author Ash Vardanian
 *
 *  This file provides two macro families for generating GEMM kernels:
 *
 *  - NK_MAKE_DOTS_VECTORS: SIMD kernels using platform-specific vector intrinsics
 *  - NK_MAKE_DOTS_SCALARS: Portable scalar kernels for any CPU
 *
 *  Both use the same B packing format (see below), enabling pack-once-use-anywhere.
 *
 *  @section packing B Matrix Packing Format
 *
 *  Computing C = A @ B^T where:
 *    - A[M, K] row-major: A[i, k] at address A + i*lda + k
 *    - B[N, K] row-major (pre-transposed): B[j, k] at address B + j*ldb + k
 *    - C[M, N] row-major: C[i, j] at address C + i*ldc + j
 *
 *  The API convention stores B as B^T for efficient SIMD access:
 *    - A[i, k:k+4] is contiguous in row-major (good)
 *    - B[j, k:k+4] is contiguous in row-major (good - already transposed)
 *
 *  Packing adds row grouping (group_size=16) for:
 *    - Zero-padding on edges (avoids boundary checks in inner loop)
 *    - Cache-friendly blocking in outer loops
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
 *  Packed as B_packed[N_padded, K] (grouped for alignment):
 *
 *    Group 0 (j=0..7, padded to 16):
 *      ┌────────────────────────────────────┐
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
 *      └────────────────────────────────────┘
 *
 *  Addressing formula for B_packed[j, k]:
 *
 *      group = j / group_size
 *      j_in_group = j % group_size
 *      B_packed[j, k] = packed[group * group_size * K + j_in_group * K + k]
 *
 *  Inner loop accesses B_packed[j, k:k+simd] which is contiguous - just ptr + k.
 */

#ifndef NK_DOTS_SERIAL_H
#define NK_DOTS_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Packed buffer header (64-byte aligned).
 *  Used by all packed matmul backends (serial, NEON, AVX-512, SVE).
 */
typedef struct {
    nk_u32_t n_groups;     // Number of column groups (ceil(n / group_size))
    nk_u32_t depth;        // K dimension (contiguous within each row)
    nk_u16_t group_size;   // Columns per group (16)
    nk_u16_t n_remainder;  // Remaining columns in last group
    nk_u32_t reserved[12]; // Padding to 64 bytes
} nk_dots_packed_header_t;

#define NK_DOTS_GROUP_SIZE_ 16 // Rows per group for alignment

/**
 *  @brief Macro to generate packed_size function for transpose-based packing.
 *
 *  Calculates buffer size needed for packed B matrix: header + N_groups * group_size * K.
 *  Edge rows are zero-padded to full group size for uniform SIMD loads.
 */
#define NK_MAKE_DOTS_PACK_SIZE(suffix, input_type, output_type)                                              \
    NK_PUBLIC nk_size_t nk_dots_##input_type##input_type##output_type##_packed_size_##suffix(nk_size_t n,   \
                                                                                             nk_size_t k) { \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;                                                   \
        nk_size_t const n_groups = (n + group_size - 1) / group_size;                                       \
        return sizeof(nk_dots_packed_header_t) + n_groups * group_size * k * sizeof(nk_##input_type##_t);   \
    }

/**
 *  @brief Macro to generate pack function for grouped row-major packing.
 *
 *  Packs B[N, K] into B_packed[N_padded, K] with row grouping.
 *  B is already in "transposed" form (N rows, K columns) as per the API convention.
 *  Each group contains group_size rows, each row has K contiguous elements.
 *  Edge groups are zero-padded to full group_size.
 */
#define NK_MAKE_DOTS_PACK(suffix, input_type, output_type)                                                         \
    NK_PUBLIC void nk_dots_##input_type##input_type##output_type##_pack_##suffix(                                  \
        nk_##input_type##_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {              \
                                                                                                                   \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;                                                          \
        nk_size_t const n_groups = (n + group_size - 1) / group_size;                                              \
                                                                                                                   \
        /* Store dimensions in header */                                                                           \
        nk_dots_packed_header_t *header = (nk_dots_packed_header_t *)b_packed;                                     \
        header->n_groups = (nk_u32_t)n_groups;                                                                     \
        header->depth = (nk_u32_t)k;                                                                               \
        header->group_size = (nk_u16_t)group_size;                                                                 \
        header->n_remainder = (nk_u16_t)(n % group_size);                                                          \
                                                                                                                   \
        nk_##input_type##_t *packed = (nk_##input_type##_t *)((char *)b_packed + sizeof(nk_dots_packed_header_t)); \
                                                                                                                   \
        /* Zero entire buffer for edge padding */                                                                  \
        nk_size_t const total_elements = n_groups * group_size * k;                                                \
        for (nk_size_t i = 0; i < total_elements; ++i) packed[i] = 0;                                              \
                                                                                                                   \
        /* Copy B[N, K] to packed[N_padded, K] with row grouping */                                                \
        /* B[j, ki] at address B + j * b_stride + ki * sizeof(element) */                                          \
        for (nk_size_t j = 0; j < n; ++j) {                                                                        \
            nk_size_t const group = j / group_size;                                                                \
            nk_size_t const j_in_group = j % group_size;                                                           \
            nk_##input_type##_t *dst_row = packed + group * group_size * k + j_in_group * k;                       \
            nk_##input_type##_t const *src_row = (nk_##input_type##_t const *)((char const *)b + j * b_stride);    \
                                                                                                                   \
            for (nk_size_t ki = 0; ki < k; ++ki) { dst_row[ki] = src_row[ki]; }                                    \
        }                                                                                                          \
    }

/**
 *  @brief SIMD GEMM macro with simplified transpose-based B packing.
 *
 *  Computes C[M, N] = A[M, K] @ B[K, N] where B is pre-packed as B_T[N, K].
 *
 *  Loop structure (GotoBLAS 5-loop design):
 *    Loop 1: NC columns at a time (L3 blocking)
 *      Loop 2: KC depth at a time (L1 blocking)
 *        Loop 3: MC rows at a time (L2 blocking)
 *          Loop 4: NR=4 columns (register tile)
 *            Loop 5: MR rows (register tile)
 *              Loop 6: k elements (SIMD accumulation) - tight inner loop
 *
 *  Inner loop structure (after B pointer setup outside k-loop):
 *
 *      b_first  = group_base + (j_in_group + 0) * depth;
 *      b_second = group_base + (j_in_group + 1) * depth;
 *      b_third  = group_base + (j_in_group + 2) * depth;
 *      b_fourth = group_base + (j_in_group + 3) * depth;
 *
 *      for (k = 0; k < aligned_depth; k += simd_width) {
 *          load_fn(b_first  + k, &b_first_vec);
 *          load_fn(b_second + k, &b_second_vec);
 *          // ... 16 FMAs for 4x4 register tile
 *      }
 */
#define NK_MAKE_DOTS_VECTORS(suffix, input_type, output_type, vec_type, state_type, init_fn, load_fn, partial_load_fn, \
                             update_fn, finalize_fn, simd_width, k_unroll, mr_size, mc_size, nc_size, kc_size)         \
                                                                                                                       \
    NK_PUBLIC void nk_dots_##suffix(nk_##input_type##_t const *a_matrix, void const *b_packed_void,                    \
                                    nk_##output_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count,       \
                                    nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {                         \
        nk_##input_type##_t const *b_packed = (nk_##input_type##_t const *)((char const *)b_packed_void +              \
                                                                            sizeof(nk_dots_packed_header_t));          \
                                                                                                                       \
        nk_size_t const register_tile_columns = 4;         /* NR: columns per finalize batch */                        \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;  /* Columns per packed group */                              \
        nk_size_t const group_stride = group_size * depth; /* Elements per group */                                    \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_idx = 0; row_idx < row_count; ++row_idx) {                                                  \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c_matrix + row_idx * c_stride);             \
            for (nk_size_t col_idx = 0; col_idx < column_count; ++col_idx) c_row[col_idx] = 0;                         \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start = 0; column_block_start < column_count; column_block_start += nc_size) {     \
            nk_size_t column_block_end = column_block_start + nc_size;                                                 \
            if (column_block_end > column_count) column_block_end = column_count;                                      \
                                                                                                                       \
            /* Loop 2: L1 cache blocking over depth */                                                                 \
            for (nk_size_t depth_block_start = 0; depth_block_start < depth; depth_block_start += kc_size) {           \
                nk_size_t depth_block_end = depth_block_start + kc_size;                                               \
                if (depth_block_end > depth) depth_block_end = depth;                                                  \
                nk_size_t const depth_block_length = depth_block_end - depth_block_start;                              \
                nk_size_t const aligned_depth = (depth_block_length / simd_width) * simd_width;                        \
                nk_size_t const remainder_depth = depth_block_length - aligned_depth;                                  \
                                                                                                                       \
                /* Loop 3: L2 cache blocking over rows */                                                              \
                for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += mc_size) {         \
                    nk_size_t row_block_end = row_block_start + mc_size;                                               \
                    if (row_block_end > row_count) row_block_end = row_count;                                          \
                                                                                                                       \
                    /* Loop 4: Register tiling over columns (4 columns per batch) */                                   \
                    for (nk_size_t tile_column_start = column_block_start; tile_column_start < column_block_end;       \
                         tile_column_start += register_tile_columns) {                                                 \
                        nk_size_t tile_column_count = register_tile_columns;                                           \
                        if (tile_column_start + tile_column_count > column_block_end)                                  \
                            tile_column_count = column_block_end - tile_column_start;                                  \
                                                                                                                       \
                        /* Compute B pointers once per column tile (outside row and k loops) */                        \
                        nk_size_t const group = tile_column_start / group_size;                                        \
                        nk_size_t const j_in_group = tile_column_start % group_size;                                   \
                        nk_##input_type##_t const *group_base = b_packed + group * group_stride;                       \
                                                                                                                       \
                        nk_##input_type##_t const *b_first = group_base + (j_in_group + 0) * depth +                   \
                                                             depth_block_start;                                        \
                        nk_##input_type##_t const *b_second = (tile_column_count > 1)                                  \
                                                                  ? group_base + (j_in_group + 1) * depth +            \
                                                                        depth_block_start                              \
                                                                  : b_first;                                           \
                        nk_##input_type##_t const *b_third = (tile_column_count > 2)                                   \
                                                                 ? group_base + (j_in_group + 2) * depth +             \
                                                                       depth_block_start                               \
                                                                 : b_first;                                            \
                        nk_##input_type##_t const *b_fourth = (tile_column_count > 3)                                  \
                                                                  ? group_base + (j_in_group + 3) * depth +            \
                                                                        depth_block_start                              \
                                                                  : b_first;                                           \
                                                                                                                       \
                        /* Loop 5: Register tiling over rows (MR rows per tile) */                                     \
                        for (nk_size_t tile_row_start = row_block_start; tile_row_start < row_block_end;               \
                             tile_row_start += mr_size) {                                                              \
                            nk_size_t tile_row_count = mr_size;                                                        \
                            if (tile_row_start + tile_row_count > row_block_end)                                       \
                                tile_row_count = row_block_end - tile_row_start;                                       \
                                                                                                                       \
                            /* Initialize MR x 4 accumulator states */                                                 \
                            state_type acc_states[mr_size][4];                                                         \
                            for (nk_size_t r = 0; r < tile_row_count; ++r) {                                           \
                                init_fn(&acc_states[r][0]);                                                            \
                                init_fn(&acc_states[r][1]);                                                            \
                                init_fn(&acc_states[r][2]);                                                            \
                                init_fn(&acc_states[r][3]);                                                            \
                            }                                                                                          \
                                                                                                                       \
                            /* A row pointers */                                                                       \
                            nk_##input_type##_t const *a_first =                                                       \
                                (nk_##input_type##_t const *)((char const *)a_matrix +                                 \
                                                              (tile_row_start + 0) * a_stride) +                       \
                                depth_block_start;                                                                     \
                            nk_##input_type##_t const *a_second =                                                      \
                                (tile_row_count > 1)                                                                   \
                                    ? (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                    (tile_row_start + 1) * a_stride) +                 \
                                          depth_block_start                                                            \
                                    : a_first;                                                                         \
                            nk_##input_type##_t const *a_third =                                                       \
                                (tile_row_count > 2)                                                                   \
                                    ? (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                    (tile_row_start + 2) * a_stride) +                 \
                                          depth_block_start                                                            \
                                    : a_first;                                                                         \
                            nk_##input_type##_t const *a_fourth =                                                      \
                                (tile_row_count > 3)                                                                   \
                                    ? (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                    (tile_row_start + 3) * a_stride) +                 \
                                          depth_block_start                                                            \
                                    : a_first;                                                                         \
                                                                                                                       \
                            /* Tight inner loop: k elements with simple ptr+k addressing */                            \
                            for (nk_size_t k = 0; k < aligned_depth; k += simd_width) {                                \
                                vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                         \
                                load_fn(a_first + k, &a_first_vec);                                                    \
                                load_fn(a_second + k, &a_second_vec);                                                  \
                                load_fn(a_third + k, &a_third_vec);                                                    \
                                load_fn(a_fourth + k, &a_fourth_vec);                                                  \
                                                                                                                       \
                                vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                         \
                                load_fn(b_first + k, &b_first_vec);                                                    \
                                load_fn(b_second + k, &b_second_vec);                                                  \
                                load_fn(b_third + k, &b_third_vec);                                                    \
                                load_fn(b_fourth + k, &b_fourth_vec);                                                  \
                                                                                                                       \
                                /* 16 FMAs: 4 A rows x 4 B columns */                                                  \
                                update_fn(&acc_states[0][0], a_first_vec, b_first_vec);                                \
                                update_fn(&acc_states[0][1], a_first_vec, b_second_vec);                               \
                                update_fn(&acc_states[0][2], a_first_vec, b_third_vec);                                \
                                update_fn(&acc_states[0][3], a_first_vec, b_fourth_vec);                               \
                                update_fn(&acc_states[1][0], a_second_vec, b_first_vec);                               \
                                update_fn(&acc_states[1][1], a_second_vec, b_second_vec);                              \
                                update_fn(&acc_states[1][2], a_second_vec, b_third_vec);                               \
                                update_fn(&acc_states[1][3], a_second_vec, b_fourth_vec);                              \
                                update_fn(&acc_states[2][0], a_third_vec, b_first_vec);                                \
                                update_fn(&acc_states[2][1], a_third_vec, b_second_vec);                               \
                                update_fn(&acc_states[2][2], a_third_vec, b_third_vec);                                \
                                update_fn(&acc_states[2][3], a_third_vec, b_fourth_vec);                               \
                                update_fn(&acc_states[3][0], a_fourth_vec, b_first_vec);                               \
                                update_fn(&acc_states[3][1], a_fourth_vec, b_second_vec);                              \
                                update_fn(&acc_states[3][2], a_fourth_vec, b_third_vec);                               \
                                update_fn(&acc_states[3][3], a_fourth_vec, b_fourth_vec);                              \
                            }                                                                                          \
                                                                                                                       \
                            /* Handle remainder k positions with partial loads */                                      \
                            if (remainder_depth > 0) {                                                                 \
                                vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                         \
                                partial_load_fn(a_first + aligned_depth, remainder_depth, &a_first_vec);               \
                                partial_load_fn(a_second + aligned_depth, remainder_depth, &a_second_vec);             \
                                partial_load_fn(a_third + aligned_depth, remainder_depth, &a_third_vec);               \
                                partial_load_fn(a_fourth + aligned_depth, remainder_depth, &a_fourth_vec);             \
                                                                                                                       \
                                vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                         \
                                partial_load_fn(b_first + aligned_depth, remainder_depth, &b_first_vec);               \
                                partial_load_fn(b_second + aligned_depth, remainder_depth, &b_second_vec);             \
                                partial_load_fn(b_third + aligned_depth, remainder_depth, &b_third_vec);               \
                                partial_load_fn(b_fourth + aligned_depth, remainder_depth, &b_fourth_vec);             \
                                                                                                                       \
                                update_fn(&acc_states[0][0], a_first_vec, b_first_vec);                                \
                                update_fn(&acc_states[0][1], a_first_vec, b_second_vec);                               \
                                update_fn(&acc_states[0][2], a_first_vec, b_third_vec);                                \
                                update_fn(&acc_states[0][3], a_first_vec, b_fourth_vec);                               \
                                update_fn(&acc_states[1][0], a_second_vec, b_first_vec);                               \
                                update_fn(&acc_states[1][1], a_second_vec, b_second_vec);                              \
                                update_fn(&acc_states[1][2], a_second_vec, b_third_vec);                               \
                                update_fn(&acc_states[1][3], a_second_vec, b_fourth_vec);                              \
                                update_fn(&acc_states[2][0], a_third_vec, b_first_vec);                                \
                                update_fn(&acc_states[2][1], a_third_vec, b_second_vec);                               \
                                update_fn(&acc_states[2][2], a_third_vec, b_third_vec);                                \
                                update_fn(&acc_states[2][3], a_third_vec, b_fourth_vec);                               \
                                update_fn(&acc_states[3][0], a_fourth_vec, b_first_vec);                               \
                                update_fn(&acc_states[3][1], a_fourth_vec, b_second_vec);                              \
                                update_fn(&acc_states[3][2], a_fourth_vec, b_third_vec);                               \
                                update_fn(&acc_states[3][3], a_fourth_vec, b_fourth_vec);                              \
                            }                                                                                          \
                                                                                                                       \
                            /* Finalize and store MR x 4 results using batched 4-way reduction */                      \
                            for (nk_size_t r = 0; r < tile_row_count; ++r) {                                           \
                                nk_##output_type##_t results[4];                                                       \
                                finalize_fn(&acc_states[r][0], &acc_states[r][1], &acc_states[r][2],                   \
                                            &acc_states[r][3], results);                                               \
                                                                                                                       \
                                nk_##output_type##_t *c_row =                                                          \
                                    (nk_##output_type##_t *)((char *)c_matrix + (tile_row_start + r) * c_stride);      \
                                for (nk_size_t c = 0; c < tile_column_count; ++c) {                                    \
                                    c_row[tile_column_start + c] += (nk_##output_type##_t)results[c];                  \
                                }                                                                                      \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Scalar GEMM macro with 4x4 register blocking.
 *
 *  Computes C[M, N] = A[M, K] @ B[K, N] where B is pre-packed as B_T[N, K].
 *
 *  Optimizations:
 *    1. Register blocking (4x4): 16 scalar accumulators stay in registers across k-loop
 *    2. K-loop unrolling (4x): reduces loop overhead, enables ILP
 *    3. A-row caching: load 4 A values once, reuse for 4 B columns
 */
#define NK_MAKE_DOTS_SCALARS(suffix, input_type, accumulator_type, output_type, load_and_convert)                   \
    NK_PUBLIC void nk_dots_##input_type##input_type##output_type##_##suffix(                                        \
        nk_##input_type##_t const *a, void const *b_packed_void, nk_##output_type##_t *c, nk_size_t m, nk_size_t n, \
        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {                                                      \
                                                                                                                    \
        nk_size_t const mr_size = 4;  /* Rows per micro-kernel */                                                   \
        nk_size_t const nr_size = 4;  /* Columns per micro-kernel */                                                \
        nk_size_t const k_unroll = 4; /* K elements per unrolled iteration */                                       \
        nk_size_t const group_size = NK_DOTS_GROUP_SIZE_;                                                           \
        nk_size_t const group_stride = group_size * k;                                                              \
                                                                                                                    \
        nk_##input_type##_t const *packed = (nk_##input_type##_t const *)((char const *)b_packed_void +             \
                                                                          sizeof(nk_dots_packed_header_t));         \
                                                                                                                    \
        /* Zero output matrix */                                                                                    \
        for (nk_size_t mi = 0; mi < m; ++mi) {                                                                      \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c + mi * c_stride);                      \
            for (nk_size_t ni = 0; ni < n; ++ni) c_row[ni] = 0;                                                     \
        }                                                                                                           \
                                                                                                                    \
        /* Process columns in groups of NR */                                                                       \
        for (nk_size_t j_block = 0; j_block < n; j_block += nr_size) {                                              \
            nk_size_t const nr_end = (j_block + nr_size < n) ? (j_block + nr_size) : n;                             \
            nk_size_t const nr_len = nr_end - j_block;                                                              \
                                                                                                                    \
            /* Compute B pointers once per column block */                                                          \
            nk_size_t const group = j_block / group_size;                                                           \
            nk_size_t const j_in_group = j_block % group_size;                                                      \
            nk_##input_type##_t const *group_base = packed + group * group_stride;                                  \
                                                                                                                    \
            nk_##input_type##_t const *b_first = group_base + (j_in_group + 0) * k;                                 \
            nk_##input_type##_t const *b_second = (nr_len > 1) ? group_base + (j_in_group + 1) * k : b_first;       \
            nk_##input_type##_t const *b_third = (nr_len > 2) ? group_base + (j_in_group + 2) * k : b_first;        \
            nk_##input_type##_t const *b_fourth = (nr_len > 3) ? group_base + (j_in_group + 3) * k : b_first;       \
                                                                                                                    \
            /* Process rows in blocks of MR */                                                                      \
            for (nk_size_t i_block = 0; i_block < m; i_block += mr_size) {                                          \
                nk_size_t const mr_end = (i_block + mr_size < m) ? (i_block + mr_size) : m;                         \
                nk_size_t const mr_len = mr_end - i_block;                                                          \
                                                                                                                    \
                /* 4x4 accumulator block */                                                                         \
                nk_##accumulator_type##_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;                               \
                nk_##accumulator_type##_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;                               \
                nk_##accumulator_type##_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;                               \
                nk_##accumulator_type##_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;                               \
                                                                                                                    \
                /* A row pointers */                                                                                \
                nk_##input_type##_t const *a_first = (nk_##input_type##_t const *)((char const *)a +                \
                                                                                   i_block * a_stride);             \
                nk_##input_type##_t const *a_second =                                                               \
                    (mr_len > 1) ? (nk_##input_type##_t const *)((char const *)a + (i_block + 1) * a_stride)        \
                                 : a_first;                                                                         \
                nk_##input_type##_t const *a_third =                                                                \
                    (mr_len > 2) ? (nk_##input_type##_t const *)((char const *)a + (i_block + 2) * a_stride)        \
                                 : a_first;                                                                         \
                nk_##input_type##_t const *a_fourth =                                                               \
                    (mr_len > 3) ? (nk_##input_type##_t const *)((char const *)a + (i_block + 3) * a_stride)        \
                                 : a_first;                                                                         \
                                                                                                                    \
                /* Main k-loop with 4x unrolling */                                                                 \
                nk_size_t ki = 0;                                                                                   \
                nk_##accumulator_type##_t a0, a1, a2, a3, b0, b1, b2, b3;                                           \
                for (; ki + k_unroll <= k; ki += k_unroll) {                                                        \
                    /* Unroll 0 */                                                                                  \
                    load_and_convert(a_first + ki, &a0), load_and_convert(a_second + ki, &a1);                      \
                    load_and_convert(a_third + ki, &a2), load_and_convert(a_fourth + ki, &a3);                      \
                    load_and_convert(b_first + ki, &b0), load_and_convert(b_second + ki, &b1);                      \
                    load_and_convert(b_third + ki, &b2), load_and_convert(b_fourth + ki, &b3);                      \
                    acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                         \
                    acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                         \
                    acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                         \
                    acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                         \
                                                                                                                    \
                    /* Unroll 1 */                                                                                  \
                    load_and_convert(a_first + ki + 1, &a0), load_and_convert(a_second + ki + 1, &a1);              \
                    load_and_convert(a_third + ki + 1, &a2), load_and_convert(a_fourth + ki + 1, &a3);              \
                    load_and_convert(b_first + ki + 1, &b0), load_and_convert(b_second + ki + 1, &b1);              \
                    load_and_convert(b_third + ki + 1, &b2), load_and_convert(b_fourth + ki + 1, &b3);              \
                    acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                         \
                    acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                         \
                    acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                         \
                    acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                         \
                                                                                                                    \
                    /* Unroll 2 */                                                                                  \
                    load_and_convert(a_first + ki + 2, &a0), load_and_convert(a_second + ki + 2, &a1);              \
                    load_and_convert(a_third + ki + 2, &a2), load_and_convert(a_fourth + ki + 2, &a3);              \
                    load_and_convert(b_first + ki + 2, &b0), load_and_convert(b_second + ki + 2, &b1);              \
                    load_and_convert(b_third + ki + 2, &b2), load_and_convert(b_fourth + ki + 2, &b3);              \
                    acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                         \
                    acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                         \
                    acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                         \
                    acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                         \
                                                                                                                    \
                    /* Unroll 3 */                                                                                  \
                    load_and_convert(a_first + ki + 3, &a0), load_and_convert(a_second + ki + 3, &a1);              \
                    load_and_convert(a_third + ki + 3, &a2), load_and_convert(a_fourth + ki + 3, &a3);              \
                    load_and_convert(b_first + ki + 3, &b0), load_and_convert(b_second + ki + 3, &b1);              \
                    load_and_convert(b_third + ki + 3, &b2), load_and_convert(b_fourth + ki + 3, &b3);              \
                    acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                         \
                    acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                         \
                    acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                         \
                    acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                         \
                }                                                                                                   \
                                                                                                                    \
                /* Remainder k-loop */                                                                              \
                for (; ki < k; ++ki) {                                                                              \
                    load_and_convert(a_first + ki, &a0), load_and_convert(a_second + ki, &a1);                      \
                    load_and_convert(a_third + ki, &a2), load_and_convert(a_fourth + ki, &a3);                      \
                    load_and_convert(b_first + ki, &b0), load_and_convert(b_second + ki, &b1);                      \
                    load_and_convert(b_third + ki, &b2), load_and_convert(b_fourth + ki, &b3);                      \
                    acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                         \
                    acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                         \
                    acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                         \
                    acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                         \
                }                                                                                                   \
                                                                                                                    \
                /* Store accumulated results */                                                                     \
                nk_##output_type##_t *c_row0 = (nk_##output_type##_t *)((char *)c + i_block * c_stride);            \
                if (nr_len > 0) c_row0[j_block] += (nk_##output_type##_t)acc00;                                     \
                if (nr_len > 1) c_row0[j_block + 1] += (nk_##output_type##_t)acc01;                                 \
                if (nr_len > 2) c_row0[j_block + 2] += (nk_##output_type##_t)acc02;                                 \
                if (nr_len > 3) c_row0[j_block + 3] += (nk_##output_type##_t)acc03;                                 \
                                                                                                                    \
                if (mr_len > 1) {                                                                                   \
                    nk_##output_type##_t *c_row1 = (nk_##output_type##_t *)((char *)c + (i_block + 1) * c_stride);  \
                    if (nr_len > 0) c_row1[j_block] += (nk_##output_type##_t)acc10;                                 \
                    if (nr_len > 1) c_row1[j_block + 1] += (nk_##output_type##_t)acc11;                             \
                    if (nr_len > 2) c_row1[j_block + 2] += (nk_##output_type##_t)acc12;                             \
                    if (nr_len > 3) c_row1[j_block + 3] += (nk_##output_type##_t)acc13;                             \
                }                                                                                                   \
                if (mr_len > 2) {                                                                                   \
                    nk_##output_type##_t *c_row2 = (nk_##output_type##_t *)((char *)c + (i_block + 2) * c_stride);  \
                    if (nr_len > 0) c_row2[j_block] += (nk_##output_type##_t)acc20;                                 \
                    if (nr_len > 1) c_row2[j_block + 1] += (nk_##output_type##_t)acc21;                             \
                    if (nr_len > 2) c_row2[j_block + 2] += (nk_##output_type##_t)acc22;                             \
                    if (nr_len > 3) c_row2[j_block + 3] += (nk_##output_type##_t)acc23;                             \
                }                                                                                                   \
                if (mr_len > 3) {                                                                                   \
                    nk_##output_type##_t *c_row3 = (nk_##output_type##_t *)((char *)c + (i_block + 3) * c_stride);  \
                    if (nr_len > 0) c_row3[j_block] += (nk_##output_type##_t)acc30;                                 \
                    if (nr_len > 1) c_row3[j_block + 1] += (nk_##output_type##_t)acc31;                             \
                    if (nr_len > 2) c_row3[j_block + 2] += (nk_##output_type##_t)acc32;                             \
                    if (nr_len > 3) c_row3[j_block + 3] += (nk_##output_type##_t)acc33;                             \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
    }

// Helper conversion functions for serial GEMM
NK_INTERNAL void nk_serial_copy_f32(nk_f32_t const *src, nk_f32_t *dst) { *dst = *src; }
NK_INTERNAL void nk_serial_copy_f64(nk_f64_t const *src, nk_f64_t *dst) { *dst = *src; }
NK_INTERNAL void nk_serial_copy_i8_to_i32(nk_i8_t const *src, nk_i32_t *dst) { *dst = (nk_i32_t)(*src); }
NK_INTERNAL void nk_serial_copy_u8_to_u32(nk_u8_t const *src, nk_u32_t *dst) { *dst = (nk_u32_t)(*src); }

// Serial implementations
NK_MAKE_DOTS_PACK_SIZE(serial, f32, f32)
NK_MAKE_DOTS_PACK(serial, f32, f32)
NK_MAKE_DOTS_SCALARS(serial, f32, f32, f32, nk_serial_copy_f32)

NK_MAKE_DOTS_PACK_SIZE(serial, f64, f64)
NK_MAKE_DOTS_PACK(serial, f64, f64)
NK_MAKE_DOTS_SCALARS(serial, f64, f64, f64, nk_serial_copy_f64)

NK_MAKE_DOTS_PACK_SIZE(serial, f16, f32)
NK_MAKE_DOTS_PACK(serial, f16, f32)
NK_MAKE_DOTS_SCALARS(serial, f16, f32, f32, nk_f16_to_f32)

NK_MAKE_DOTS_PACK_SIZE(serial, bf16, f32)
NK_MAKE_DOTS_PACK(serial, bf16, f32)
NK_MAKE_DOTS_SCALARS(serial, bf16, f32, f32, nk_bf16_to_f32)

NK_MAKE_DOTS_PACK_SIZE(serial, i8, i32)
NK_MAKE_DOTS_PACK(serial, i8, i32)
NK_MAKE_DOTS_SCALARS(serial, i8, i32, i32, nk_serial_copy_i8_to_i32)

NK_MAKE_DOTS_PACK_SIZE(serial, u8, u32)
NK_MAKE_DOTS_PACK(serial, u8, u32)
NK_MAKE_DOTS_SCALARS(serial, u8, u32, u32, nk_serial_copy_u8_to_u32)

/*  BF16 compact: truncate F32 -> BF16 in-place.
 *  Reads F32 matrix with c_stride, writes BF16 tightly packed (stride = n * sizeof(bf16)).
 */
NK_PUBLIC void nk_dots_bf16bf16bf16_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride) {
    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_f32_t const *src_row = c_f32 + row * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row * n;
        for (nk_size_t col = 0; col < n; col++) { nk_f32_to_bf16(src_row + col, dst_row + col); }
    }
}

/*  I8 compact: re-normalize I32 -> I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 / sqrt(a_norm[i] * b_norm[j])
 *  Output is tightly packed (stride = n * sizeof(i8)).
 */
NK_PUBLIC void nk_dots_i8i8i8_serial(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride,
                                     nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms) {
    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_i32_t const *src_row = c_i32 + row * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row * n;

        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row];
        nk_f32_t a_rsqrt = (a_norm_f32 > 0) ? (1.0f / NK_F32_SQRT(a_norm_f32)) : 0.0f;

        for (nk_size_t col = 0; col < n; col++) {
            nk_f32_t b_norm_f32 = (nk_f32_t)b_squared_norms[col];
            nk_f32_t b_rsqrt = (b_norm_f32 > 0) ? (1.0f / NK_F32_SQRT(b_norm_f32)) : 0.0f;

            nk_f32_t normalized = (nk_f32_t)src_row[col] * 127.0f * a_rsqrt * b_rsqrt;
            nk_i32_t clamped = (nk_i32_t)normalized;
            if (clamped < -128) clamped = -128;
            if (clamped > 127) clamped = 127;
            dst_row[col] = (nk_i8_t)clamped;
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOTS_SERIAL_H
