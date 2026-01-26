/**
 *  @brief SIMD-accelerated GEMM for `f32`/`f64` using ARM SME with `f64` accumulators.
 *  @file include/numkong/dots/smef64.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  Uses ARM SME with `FEAT_SME_F64F64` for high-precision GEMM.
 *  Requires Apple M4 or equivalent with `f64` outer product support.
 *
 *  Provides `f32` and `f64` GEMM using `ZA64` tiles:
 *  - `f32` inputs with `f64` accumulation: higher precision than `ZA32`
 *  - Native `f64` GEMM with Kahan compensated summation
 *
 *  Tile dimensions for SVL=512 (Apple M4):
 *  - `ZA64` tile: 8 × 8 `f64` elements (512B)
 *  - `f64` vectors: 8 elements per SVE vector
 *  - `f32` vectors: 16 elements per SVE vector, converted to `f64`
 *
 *  Key instructions:
 *  - `svmopa_za64_f64_m` / `FMOPA`: `f64` outer product, 16cy amortized
 *  - `svcvt_f64_f32_x` / `FCVT`: `f32` → `f64` conversion
 */
#ifndef NK_DOTS_SMEF64_H
#define NK_DOTS_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/dots/sme.h" // For nk_dots_sme_packed_header_t

#include <arm_sme.h>
#include <arm_sve.h>

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Single-Precision Floats (f32)

/**
 *  @brief Returns packed buffer size in bytes for `f32` B matrix.
 *
 *  Layout uses `ZA64` tile dimensions (8×8 for 512-bit SVL).
 *  `f32` inputs are converted to `f64` during computation for higher precision.
 *
 *  @param column_count Number of rows in B (output columns).
 *  @param depth Number of columns in B (shared dimension).
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw(); // `f32` depth elements per tile (16 for SVL=512)

    nk_size_t const column_tile_count = (columns + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f32_t);

    return size;
}

/**
 *  @brief Packs `f32` B matrix into SME-optimized layout.
 *
 *  @param b Input B matrix in row-major order.
 *  @param column_count Number of rows in B (output columns).
 *  @param depth Number of columns in B (shared dimension).
 *  @param b_stride Row stride in bytes for B.
 *  @param b_packed Output packed buffer from `nk_dots_packed_size_f32_smef64`.
 */
NK_PUBLIC void nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const tile_dimension = svcntsd();                       // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw();                // `f32` depth elements per tile (16 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size; // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const column_tile_count = (columns + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb(); // streaming vector length in bytes

    nk_f32_t *tiles = (nk_f32_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0f;

    // Pack data into tiles with column-major layout within each tile
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tile_count + depth_tile_idx;
            nk_f32_t *tile_output = tiles + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tile_dimension;
            nk_size_t const src_col_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= columns) ? tile_dimension : (columns - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= depth) ? depth_tile_size
                                                                                      : (depth - src_col_start);

            for (nk_size_t row_idx = 0; row_idx < rows_to_pack; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < cols_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_col_start + col_idx;
                    nk_size_t const dst_idx = col_idx * tile_dimension + row_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_f32_sme_kernel_(nk_f32_t const *a, void const *b_packed,
                                                                            nk_f32_t *c, nk_size_t rows,
                                                                            nk_size_t columns, nk_size_t depth,
                                                                            nk_size_t a_stride_elements,
                                                                            nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();        // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_double = svptrue_b64();

    // Process all row tiles (including partial)
    for (nk_size_t row_tile_idx = 0; row_tile_idx < (rows + tile_dimension - 1) / tile_dimension; row_tile_idx++) {
        nk_size_t const row_start = row_tile_idx * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b64((uint64_t)0, rows_remaining);

        nk_size_t column_tile_idx = 0;

        // Process 4 column tiles at a time using ZA0-ZA3
        for (; column_tile_idx + 4 <= column_tile_count; column_tile_idx += 4) {
            svzero_za();

            // Accumulate over all depth tiles
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;
                nk_size_t const b_tile_idx0 = (column_tile_idx + 0) * depth_tile_count + depth_tile_idx;
                nk_size_t const b_tile_idx1 = (column_tile_idx + 1) * depth_tile_count + depth_tile_idx;
                nk_size_t const b_tile_idx2 = (column_tile_idx + 2) * depth_tile_count + depth_tile_idx;
                nk_size_t const b_tile_idx3 = (column_tile_idx + 3) * depth_tile_count + depth_tile_idx;
                nk_f32_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                nk_f32_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                nk_f32_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                nk_f32_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                // Process depth in tile_dimension chunks (8 f64 elements at a time)
                for (nk_size_t depth_sub = 0; depth_sub < depth_tile_size; depth_sub += tile_dimension) {
                    nk_size_t const depth_absolute_offset = depth_offset + depth_sub;
                    nk_size_t const depth_remaining = (depth_absolute_offset + tile_dimension <= depth)
                                                          ? tile_dimension
                                                          : ((depth_absolute_offset < depth) ? (depth - depth_absolute_offset) : 0);
                    svbool_t const predicate_valid_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f32_t const *pointer_a = a + (row_start + row_idx) * a_stride_elements + depth_absolute_offset;
                        svfloat32_t a_f32 = svld1_f32(svwhilelt_b32((uint64_t)0, depth_remaining), pointer_a);
                        svfloat64_t vector_a = svcvt_f64_f32_x(predicate_valid_depth, a_f32);

                        nk_f32_t const *b_col0 = b_tile0 + (depth_sub + row_idx) * tile_dimension;
                        nk_f32_t const *b_col1 = b_tile1 + (depth_sub + row_idx) * tile_dimension;
                        nk_f32_t const *b_col2 = b_tile2 + (depth_sub + row_idx) * tile_dimension;
                        nk_f32_t const *b_col3 = b_tile3 + (depth_sub + row_idx) * tile_dimension;
                        svfloat64_t vector_b_tile_0 = svcvt_f64_f32_x(predicate_double, svld1_f32(predicate_double, b_col0));
                        svfloat64_t vector_b_tile_1 = svcvt_f64_f32_x(predicate_double, svld1_f32(predicate_double, b_col1));
                        svfloat64_t vector_b_tile_2 = svcvt_f64_f32_x(predicate_double, svld1_f32(predicate_double, b_col2));
                        svfloat64_t vector_b_tile_3 = svcvt_f64_f32_x(predicate_double, svld1_f32(predicate_double, b_col3));

                        svmopa_za64_f64_m(0, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_0);
                        svmopa_za64_f64_m(1, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_1);
                        svmopa_za64_f64_m(2, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_2);
                        svmopa_za64_f64_m(3, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_3);
                    }
                }
            }

            // Store results from all 4 ZA tiles
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_size_t const col_start0 = (column_tile_idx + 0) * tile_dimension;
                nk_size_t const col_start1 = (column_tile_idx + 1) * tile_dimension;
                nk_size_t const col_start2 = (column_tile_idx + 2) * tile_dimension;
                nk_size_t const col_start3 = (column_tile_idx + 3) * tile_dimension;

                // Direct extraction via buffer (svread_hor not available)
                nk_f64_t za_buf[8] __attribute__((aligned(64)));

                svst1_hor_za64(0, row_idx, predicate_double, za_buf);
                svst1_f32(predicate_double, c + (row_start + row_idx) * c_stride_elements + col_start0,
                          svcvt_f32_f64_x(predicate_double, svld1_f64(predicate_double, za_buf)));

                svst1_hor_za64(1, row_idx, predicate_double, za_buf);
                svst1_f32(predicate_double, c + (row_start + row_idx) * c_stride_elements + col_start1,
                          svcvt_f32_f64_x(predicate_double, svld1_f64(predicate_double, za_buf)));

                svst1_hor_za64(2, row_idx, predicate_double, za_buf);
                svst1_f32(predicate_double, c + (row_start + row_idx) * c_stride_elements + col_start2,
                          svcvt_f32_f64_x(predicate_double, svld1_f64(predicate_double, za_buf)));

                svst1_hor_za64(3, row_idx, predicate_double, za_buf);
                svst1_f32(predicate_double, c + (row_start + row_idx) * c_stride_elements + col_start3,
                          svcvt_f32_f64_x(predicate_double, svld1_f64(predicate_double, za_buf)));
            }
        }

        // Process remaining column tiles one at a time
        for (; column_tile_idx < column_tile_count; column_tile_idx++) {
            nk_size_t const col_start = column_tile_idx * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b64((uint64_t)0, cols_remaining);

            svzero_za();

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;
                nk_size_t const b_tile_idx = column_tile_idx * depth_tile_count + depth_tile_idx;
                nk_f32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t depth_sub = 0; depth_sub < depth_tile_size; depth_sub += tile_dimension) {
                    nk_size_t const depth_absolute_offset = depth_offset + depth_sub;
                    nk_size_t const depth_remaining = (depth_absolute_offset + tile_dimension <= depth)
                                                          ? tile_dimension
                                                          : ((depth_absolute_offset < depth) ? (depth - depth_absolute_offset) : 0);
                    svbool_t const predicate_valid_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f32_t const *pointer_a = a + (row_start + row_idx) * a_stride_elements + depth_absolute_offset;
                        svfloat32_t a_f32 = svld1_f32(svwhilelt_b32((uint64_t)0, depth_remaining), pointer_a);
                        svfloat64_t vector_a = svcvt_f64_f32_x(predicate_valid_depth, a_f32);

                        nk_f32_t const *b_col = b_tile + (depth_sub + row_idx) * tile_dimension;
                        svfloat32_t b_f32 = svld1_f32(predicate_double, b_col);
                        svfloat64_t vector_b = svcvt_f64_f32_x(predicate_double, b_f32);

                        svmopa_za64_f64_m(0, predicate_valid_rows, predicate_valid_columns, vector_a, vector_b);
                    }
                }
            }

            // Store results with predication
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t za_buf[8] __attribute__((aligned(64)));
                svst1_hor_za64(0, row_idx, predicate_double, za_buf);
                svfloat64_t zvector_a = svld1_f64(predicate_double, za_buf);
                svfloat32_t c_vec = svcvt_f32_f64_x(predicate_double, zvector_a);
                nk_f32_t *c_row = c + (row_start + row_idx) * c_stride_elements + col_start;
                svst1_f32(predicate_valid_columns, c_row, c_vec);
            }
        }
    }
}

/**
 *  @brief Computes C = A × Bᵀ using packed `f32` B, accumulating with `f64` precision.
 *
 *  High-precision matrix multiplication using `f64` accumulators for scientific computing.
 *  All edge handling is done via predicates in the kernel.
 *
 *  @param a Input matrix A (M × K), row-major.
 *  @param b_packed Pre-packed B matrix from `nk_dots_pack_f32_smef64`.
 *  @param c Output matrix C (M × N), row-major, `f32`.
 *  @param rows Number of rows in A and C.
 *  @param columns Number of columns in C (rows in original B).
 *  @param depth Shared dimension (columns in A, columns in original B).
 *  @param a_stride Byte stride between rows of A.
 *  @param c_stride Byte stride between rows of C.
 */
NK_PUBLIC void nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_f32_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion

#pragma region Double-Precision Floats (f64)

/**
 *  @brief Returns packed buffer size in bytes for `f64` B matrix.
 *
 *  @param column_count Number of rows in B (output columns).
 *  @param depth Number of columns in B (shared dimension).
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)

    nk_size_t const column_tile_count = (columns + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f64_t);

    return size;
}

/**
 *  @brief Packs `f64` B matrix into SME-optimized layout.
 *
 *  @param b Input B matrix in row-major order.
 *  @param column_count Number of rows in B (output columns).
 *  @param depth Number of columns in B (shared dimension).
 *  @param b_stride Row stride in bytes for B.
 *  @param b_packed Output packed buffer from `nk_dots_packed_size_f64_smef64`.
 */
NK_PUBLIC void nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const tile_dimension = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);

    nk_size_t const column_tile_count = (columns + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb(); // streaming vector length in bytes

    nk_f64_t *tiles = (nk_f64_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0;

    // Pack data into tiles with column-major layout within each tile
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tile_count + depth_tile_idx;
            nk_f64_t *tile_output = tiles + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tile_dimension;
            nk_size_t const src_col_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= columns) ? tile_dimension : (columns - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= depth) ? depth_tile_size
                                                                                      : (depth - src_col_start);

            for (nk_size_t row_idx = 0; row_idx < rows_to_pack; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < cols_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_col_start + col_idx;
                    nk_size_t const dst_idx = col_idx * tile_dimension + row_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

// Batch size for Kahan compensation: accumulate this many depth-tiles before extracting
#define NK_KAHAN_BATCH_SIZE 32

__arm_locally_streaming __arm_new("za") static void nk_dots_f64_sme_kernel_(nk_f64_t const *a, void const *b_packed,
                                                                            nk_f64_t *c, nk_size_t rows,
                                                                            nk_size_t columns, nk_size_t depth,
                                                                            nk_size_t a_stride_elements,
                                                                            nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();        // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntd(); // 8 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_f64_t const *b_tiles = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_double = svptrue_b64();

    // Kahan accumulator and compensation arrays - 4 sets for 4-tile processing
    nk_f64_t partial_sum[4][64];
    nk_f64_t compensation[4][64];
    nk_f64_t accumulator[4][64];

    nk_size_t const batch_size = NK_KAHAN_BATCH_SIZE < depth_tile_count ? NK_KAHAN_BATCH_SIZE : depth_tile_count;

    // Process all row tiles (including partial)
    for (nk_size_t row_tile_idx = 0; row_tile_idx < (rows + tile_dimension - 1) / tile_dimension; row_tile_idx++) {
        nk_size_t const row_start = row_tile_idx * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b64((uint64_t)0, rows_remaining);

        // Process 4 column tiles at a time using ZA0-ZA3
        nk_size_t column_tile_index = 0;
        for (; column_tile_index + 4 <= column_tile_count; column_tile_index += 4) {
            nk_size_t const col_start0 = column_tile_index * tile_dimension;
            nk_size_t const col_start1 = (column_tile_index + 1) * tile_dimension;
            nk_size_t const col_start2 = (column_tile_index + 2) * tile_dimension;
            nk_size_t const col_start3 = (column_tile_index + 3) * tile_dimension;

            // Initialize all 4 Kahan accumulators
            svfloat64_t zero_vec = svdup_f64(0.0);
            for (int t = 0; t < 4; t++) {
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_f64(predicate_double, accumulator[t] + row_idx * tile_dimension, zero_vec);
                    svst1_f64(predicate_double, compensation[t] + row_idx * tile_dimension, zero_vec);
                }
            }

            // Process depth tiles in batches for Kahan compensation
            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_count;
                 depth_batch_start += batch_size) {
                nk_size_t const depth_batch_end = (depth_batch_start + batch_size < depth_tile_count)
                                                      ? depth_batch_start + batch_size
                                                      : depth_tile_count;

                svzero_za();

                // Accumulate within this batch - all 4 column tiles
                for (nk_size_t depth_tile_idx = depth_batch_start; depth_tile_idx < depth_batch_end; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;
                    nk_size_t const depth_remaining = (depth_offset + depth_tile_size <= depth)
                                                          ? depth_tile_size
                                                          : ((depth_offset < depth) ? (depth - depth_offset) : 0);
                    svbool_t const predicate_valid_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    // Get B tile pointers for all 4 column tiles
                    nk_size_t const b_tile_idx0 = column_tile_index * depth_tile_count + depth_tile_idx;
                    nk_size_t const b_tile_idx1 = (column_tile_index + 1) * depth_tile_count + depth_tile_idx;
                    nk_size_t const b_tile_idx2 = (column_tile_index + 2) * depth_tile_count + depth_tile_idx;
                    nk_size_t const b_tile_idx3 = (column_tile_index + 3) * depth_tile_count + depth_tile_idx;
                    nk_f64_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                    nk_f64_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                    nk_f64_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                    nk_f64_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                    // Outer product accumulation to all 4 ZA tiles
                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f64_t const *pointer_a = a + (row_start + row_idx) * a_stride_elements + depth_offset;
                        svfloat64_t vector_a = svld1_f64(predicate_valid_depth, pointer_a);

                        svfloat64_t vector_b_tile_0 = svld1_f64(predicate_double, b_tile0 + row_idx * depth_tile_size);
                        svfloat64_t vector_b_tile_1 = svld1_f64(predicate_double, b_tile1 + row_idx * depth_tile_size);
                        svfloat64_t vector_b_tile_2 = svld1_f64(predicate_double, b_tile2 + row_idx * depth_tile_size);
                        svfloat64_t vector_b_tile_3 = svld1_f64(predicate_double, b_tile3 + row_idx * depth_tile_size);

                        svmopa_za64_f64_m(0, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_0);
                        svmopa_za64_f64_m(1, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_1);
                        svmopa_za64_f64_m(2, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_2);
                        svmopa_za64_f64_m(3, predicate_valid_rows, predicate_double, vector_a, vector_b_tile_3);
                    }
                }

                // Extract partial sums from all 4 ZA tiles
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_hor_za64(0, row_idx, predicate_double, partial_sum[0] + row_idx * tile_dimension);
                    svst1_hor_za64(1, row_idx, predicate_double, partial_sum[1] + row_idx * tile_dimension);
                    svst1_hor_za64(2, row_idx, predicate_double, partial_sum[2] + row_idx * tile_dimension);
                    svst1_hor_za64(3, row_idx, predicate_double, partial_sum[3] + row_idx * tile_dimension);
                }

                // Apply Kahan compensation to all 4 tiles
                for (int t = 0; t < 4; t++) {
                    for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                        nk_size_t const base_idx = row_idx * tile_dimension;

                        svfloat64_t accumulator_vector = svld1_f64(predicate_double, accumulator[t] + base_idx);
                        svfloat64_t compensation_vector = svld1_f64(predicate_double, compensation[t] + base_idx);
                        svfloat64_t partial_vector = svld1_f64(predicate_double, partial_sum[t] + base_idx);

                        // Kahan summation: y = part - comp; t = acc + y; comp = (t - acc) - y; acc = t
                        svfloat64_t y = svsub_f64_x(predicate_double, partial_vector, compensation_vector);
                        svfloat64_t sum = svadd_f64_x(predicate_double, accumulator_vector, y);
                        compensation_vector = svsub_f64_x(predicate_double, svsub_f64_x(predicate_double, sum, accumulator_vector), y);
                        accumulator_vector = sum;

                        svst1_f64(predicate_double, accumulator[t] + base_idx, accumulator_vector);
                        svst1_f64(predicate_double, compensation[t] + base_idx, compensation_vector);
                    }
                }
            }

            // Store results from all 4 tiles
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements;
                svst1_f64(predicate_double, c_row + col_start0, svld1_f64(predicate_double, accumulator[0] + row_idx * tile_dimension));
                svst1_f64(predicate_double, c_row + col_start1, svld1_f64(predicate_double, accumulator[1] + row_idx * tile_dimension));
                svst1_f64(predicate_double, c_row + col_start2, svld1_f64(predicate_double, accumulator[2] + row_idx * tile_dimension));
                svst1_f64(predicate_double, c_row + col_start3, svld1_f64(predicate_double, accumulator[3] + row_idx * tile_dimension));
            }
        }

        // Process remaining column tiles (0-3 tiles) one at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b64((uint64_t)0, cols_remaining);

            // Initialize Kahan accumulators using SVE
            svfloat64_t zero_vec = svdup_f64(0.0);
            for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                svst1_f64(predicate_double, accumulator[0] + row_idx * tile_dimension, zero_vec);
                svst1_f64(predicate_double, compensation[0] + row_idx * tile_dimension, zero_vec);
            }

            // Process depth tiles in batches for Kahan compensation
            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_count;
                 depth_batch_start += batch_size) {
                nk_size_t const depth_batch_end = (depth_batch_start + batch_size < depth_tile_count)
                                                      ? depth_batch_start + batch_size
                                                      : depth_tile_count;

                svzero_za();

                // Accumulate within this batch
                for (nk_size_t depth_tile_idx = depth_batch_start; depth_tile_idx < depth_batch_end; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;
                    nk_size_t const depth_remaining = (depth_offset + depth_tile_size <= depth)
                                                          ? depth_tile_size
                                                          : ((depth_offset < depth) ? (depth - depth_offset) : 0);
                    svbool_t const predicate_valid_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_idx;
                    nk_f64_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                    // Outer product accumulation
                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f64_t const *pointer_a = a + (row_start + row_idx) * a_stride_elements + depth_offset;
                        svfloat64_t vector_a = svld1_f64(predicate_valid_depth, pointer_a);

                        nk_f64_t const *b_col = b_tile + row_idx * depth_tile_size;
                        svfloat64_t vector_b = svld1_f64(predicate_double, b_col);

                        svmopa_za64_f64_m(0, predicate_valid_rows, predicate_valid_columns, vector_a, vector_b);
                    }
                }

                // Extract partial sums from ZA tile
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_hor_za64(0, row_idx, predicate_double, partial_sum[0] + row_idx * tile_dimension);
                }

                // Apply Kahan compensation
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    nk_size_t const base_idx = row_idx * tile_dimension;

                    svfloat64_t accumulator_vector = svld1_f64(predicate_double, accumulator[0] + base_idx);
                    svfloat64_t compensation_vector = svld1_f64(predicate_double, compensation[0] + base_idx);
                    svfloat64_t partial_vector = svld1_f64(predicate_double, partial_sum[0] + base_idx);

                    // Kahan summation: y = part - comp; t = acc + y; comp = (t - acc) - y; acc = t
                    svfloat64_t y = svsub_f64_x(predicate_double, partial_vector, compensation_vector);
                    svfloat64_t sum = svadd_f64_x(predicate_double, accumulator_vector, y);
                    compensation_vector = svsub_f64_x(predicate_double, svsub_f64_x(predicate_double, sum, accumulator_vector), y);
                    accumulator_vector = sum;

                    svst1_f64(predicate_double, accumulator[0] + base_idx, accumulator_vector);
                    svst1_f64(predicate_double, compensation[0] + base_idx, compensation_vector);
                }
            }

            // Store results with predication
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements + col_start;
                svfloat64_t accumulator_vectorec = svld1_f64(predicate_double, accumulator[0] + row_idx * tile_dimension);
                svst1_f64(predicate_valid_columns, c_row, accumulator_vectorec);
            }
        }
    }
}

/**
 *  @brief Computes C = A × Bᵀ using packed `f64` B with Kahan compensated summation.
 *
 *  High-precision `f64` GEMM with ~16-17 decimal digits of precision.
 *  All edge handling is done via predicates in the kernel.
 *
 *  @param a Input matrix A (M × K), row-major.
 *  @param b_packed Pre-packed B matrix from `nk_dots_pack_f64_smef64`.
 *  @param c Output matrix C (M × N), row-major.
 *  @param rows Number of rows in A and C.
 *  @param columns Number of columns in C (rows in original B).
 *  @param depth Shared dimension (columns in A, columns in original B).
 *  @param a_stride Byte stride between rows of A.
 *  @param c_stride Byte stride between rows of C.
 */
NK_PUBLIC void nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_dots_f64_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion

#pragma region Symmetric GEMM

__arm_locally_streaming static void nk_dots_symmetric_f32_smef64_kernel_(
    nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements,
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length_d = svcntd(); // f64 elements per vector (half of f32)
    svbool_t const predicate_double = svptrue_b64();

    nk_size_t const row_end = row_start + row_count;

    // Compute specified rows of symmetric matrix
    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_f32_t const *pointer_row_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_f32_t const *pointer_row_j = vectors + j * stride_elements;

            // SVE vectorized dot product with f64 accumulation
            svfloat64_t accumulator = svdup_f64(0.0);
            nk_size_t depth_index = 0;

            // Process in chunks of vector_length_d f32 elements (converts to f64)
            while (depth_index + vector_length_d <= depth) {
                svbool_t predicate_single = svwhilelt_b32((uint32_t)0, (uint32_t)vector_length_d);
                svfloat32_t vector_i_f32 = svld1_f32(predicate_single, pointer_row_i + depth_index);
                svfloat32_t vector_j_f32 = svld1_f32(predicate_single, pointer_row_j + depth_index);
                svfloat64_t vector_i = svcvt_f64_f32_x(predicate_double, vector_i_f32);
                svfloat64_t vector_j = svcvt_f64_f32_x(predicate_double, vector_j_f32);
                accumulator = svmla_f64_x(predicate_double, accumulator, vector_i, vector_j);
                depth_index += vector_length_d;
            }
            // Handle remainder
            if (depth_index < depth) {
                nk_size_t remaining = depth - depth_index;
                svbool_t predicate_single_tail = svwhilelt_b32((uint32_t)0, (uint32_t)remaining);
                svbool_t predicate_double_tail = svwhilelt_b64((uint64_t)0, remaining);
                svfloat32_t vector_i_f32 = svld1_f32(predicate_single_tail, pointer_row_i + depth_index);
                svfloat32_t vector_j_f32 = svld1_f32(predicate_single_tail, pointer_row_j + depth_index);
                svfloat64_t vector_i = svcvt_f64_f32_x(predicate_double_tail, vector_i_f32);
                svfloat64_t vector_j = svcvt_f64_f32_x(predicate_double_tail, vector_j_f32);
                accumulator = svmla_f64_x(predicate_double_tail, accumulator, vector_i, vector_j);
            }
            nk_f32_t dot = (nk_f32_t)svaddv_f64(predicate_double, accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f32_smef64_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
}

__arm_locally_streaming static void nk_dots_symmetric_f64_smef64_kernel_(
    nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements,
    nk_f64_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcntd(); // f64 elements per vector
    svbool_t const predicate_double = svptrue_b64();

    nk_size_t const row_end = row_start + row_count;

    // Compute specified rows of symmetric matrix
    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_f64_t const *pointer_row_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_f64_t const *pointer_row_j = vectors + j * stride_elements;

            // SVE vectorized dot product
            svfloat64_t accumulator = svdup_f64(0.0);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svfloat64_t vector_i = svld1_f64(predicate_double, pointer_row_i + depth_index);
                svfloat64_t vector_j = svld1_f64(predicate_double, pointer_row_j + depth_index);
                accumulator = svmla_f64_x(predicate_double, accumulator, vector_i, vector_j);
                depth_index += vector_length;
            }
            // Handle remainder
            if (depth_index < depth) {
                svbool_t predicate_tail = svwhilelt_b64((uint64_t)depth_index, (uint64_t)depth);
                svfloat64_t vector_i = svld1_f64(predicate_tail, pointer_row_i + depth_index);
                svfloat64_t vector_j = svld1_f64(predicate_tail, pointer_row_j + depth_index);
                accumulator = svmla_f64_x(predicate_tail, accumulator, vector_i, vector_j);
            }
            nk_f64_t dot = svaddv_f64(predicate_double, accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_smef64_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
}

#pragma endregion

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SMEF64_H
