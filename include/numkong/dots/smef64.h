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
    nk_size_t const tile_dim = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw(); // `f32` depth elements per tile (16 for SVL=512)

    nk_size_t const column_tile_count = (columns + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dim * depth_tile_size * sizeof(nk_f32_t);

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

    nk_size_t const tile_dim = svcntsd();                       // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw();                // `f32` depth elements per tile (16 for SVL=512)
    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const column_tile_count = (columns + tile_dim - 1) / tile_dim;
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

            nk_size_t const src_row_start = column_tile_idx * tile_dim;
            nk_size_t const src_col_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= columns) ? tile_dim : (columns - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= depth) ? depth_tile_size
                                                                                      : (depth - src_col_start);

            for (nk_size_t row_idx = 0; row_idx < rows_to_pack; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < cols_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_col_start + col_idx;
                    nk_size_t const dst_idx = col_idx * tile_dim + row_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/*  `f32` GEMM core kernel using SME `f64` outer products with predication.
 *  Processes all tiles including partial edge tiles using SVE predicates.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f32_sme_kernel_(nk_f32_t const *a, void const *b_packed,
                                                                            nk_f32_t *c, nk_size_t rows,
                                                                            nk_size_t columns, nk_size_t depth,
                                                                            nk_size_t a_stride_elements,
                                                                            nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntd();        // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_d = svptrue_b64();

    // Process all row tiles (including partial)
    for (nk_size_t row_tile_idx = 0; row_tile_idx < (rows + tile_dim - 1) / tile_dim; row_tile_idx++) {
        nk_size_t const row_start = row_tile_idx * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b64((uint64_t)0, rows_remaining);

        // Process all column tiles (including partial)
        for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
            nk_size_t const col_start = column_tile_idx * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b64((uint64_t)0, cols_remaining);

            svzero_za();

            // Accumulate over all depth tiles
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;
                nk_size_t const b_tile_idx = column_tile_idx * depth_tile_count + depth_tile_idx;
                nk_f32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process depth in tile_dim chunks (8 f64 elements at a time)
                for (nk_size_t depth_sub = 0; depth_sub < depth_tile_size; depth_sub += tile_dim) {
                    nk_size_t const d_abs = depth_offset + depth_sub;
                    nk_size_t const depth_remaining = (d_abs + tile_dim <= depth)
                                                          ? tile_dim
                                                          : ((d_abs < depth) ? (depth - d_abs) : 0);
                    svbool_t const pred_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    // Outer product accumulation (use `rows_remaining` instead of break)
                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f32_t const *a_ptr = a + (row_start + row_idx) * a_stride_elements + d_abs;
                        svfloat32_t a_f32 = svld1_f32(svwhilelt_b32((uint64_t)0, depth_remaining), a_ptr);
                        svfloat64_t a_vec = svcvt_f64_f32_x(pred_depth, a_f32);

                        nk_f32_t const *b_col = b_tile + (depth_sub + row_idx) * tile_dim;
                        svfloat32_t b_f32 = svld1_f32(ptrue_d, b_col);
                        svfloat64_t b_vec = svcvt_f64_f32_x(ptrue_d, b_f32);

                        svmopa_za64_f64_m(0, pred_rows, pred_cols, a_vec, b_vec);
                    }
                }
            }

            // Store results: extract f64 from ZA, convert to f32, store with predication
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t za_buf[8] __attribute__((aligned(64)));
                svst1_hor_za64(0, row_idx, ptrue_d, za_buf);
                svfloat64_t za_vec = svld1_f64(ptrue_d, za_buf);
                svfloat32_t c_vec = svcvt_f32_f64_x(ptrue_d, za_vec);
                nk_f32_t *c_row = c + (row_start + row_idx) * c_stride_elements + col_start;
                svst1_f32(pred_cols, c_row, c_vec);
            }
        }
    }
}

/**
 *  @brief Computes C = A × B^T using packed `f32` B, accumulating with `f64` precision.
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

/**
 *  @brief Returns packed buffer size in bytes for `f64` B matrix.
 *
 *  @param column_count Number of rows in B (output columns).
 *  @param depth Number of columns in B (shared dimension).
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dim = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)

    nk_size_t const column_tile_count = (columns + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth + depth_tile_size - 1) / depth_tile_size;

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dim * depth_tile_size * sizeof(nk_f64_t);

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

    nk_size_t const tile_dim = svcntsd();        // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);

    nk_size_t const column_tile_count = (columns + tile_dim - 1) / tile_dim;
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

            nk_size_t const src_row_start = column_tile_idx * tile_dim;
            nk_size_t const src_col_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= columns) ? tile_dim : (columns - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= depth) ? depth_tile_size
                                                                                      : (depth - src_col_start);

            for (nk_size_t row_idx = 0; row_idx < rows_to_pack; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < cols_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_col_start + col_idx;
                    nk_size_t const dst_idx = col_idx * tile_dim + row_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

// Batch size for Kahan compensation: accumulate this many depth-tiles before extracting
#define NK_KAHAN_BATCH_SIZE 32

/*  `f64` GEMM core kernel using SME `f64` outer products with Kahan compensated summation.
 *  Processes all tiles including partial edge tiles using SVE predicates.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f64_sme_kernel_(nk_f64_t const *a, void const *b_packed,
                                                                            nk_f64_t *c, nk_size_t rows,
                                                                            nk_size_t columns, nk_size_t depth,
                                                                            nk_size_t a_stride_elements,
                                                                            nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntd();        // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntd(); // 8 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_f64_t const *b_tiles = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_d = svptrue_b64();

    // Kahan accumulator and compensation arrays (one tile's worth)
    nk_f64_t partial[64];
    nk_f64_t comp[64];
    nk_f64_t acc[64];

    nk_size_t const batch_size = NK_KAHAN_BATCH_SIZE < depth_tile_count ? NK_KAHAN_BATCH_SIZE : depth_tile_count;

    // Process all row tiles (including partial)
    for (nk_size_t row_tile_idx = 0; row_tile_idx < (rows + tile_dim - 1) / tile_dim; row_tile_idx++) {
        nk_size_t const row_start = row_tile_idx * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b64((uint64_t)0, rows_remaining);

        // Process all column tiles (including partial)
        for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
            nk_size_t const col_start = column_tile_idx * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b64((uint64_t)0, cols_remaining);

            // Initialize Kahan accumulators using SVE
            svfloat64_t zero_vec = svdup_f64(0.0);
            for (nk_size_t row_idx = 0; row_idx < tile_dim; row_idx++) {
                svst1_f64(ptrue_d, acc + row_idx * tile_dim, zero_vec);
                svst1_f64(ptrue_d, comp + row_idx * tile_dim, zero_vec);
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
                    svbool_t const pred_depth = svwhilelt_b64((uint64_t)0, depth_remaining);

                    nk_size_t const b_tile_idx = column_tile_idx * depth_tile_count + depth_tile_idx;
                    nk_f64_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                    // Outer product accumulation
                    for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                        nk_f64_t const *a_ptr = a + (row_start + row_idx) * a_stride_elements + depth_offset;
                        svfloat64_t a_vec = svld1_f64(pred_depth, a_ptr);

                        nk_f64_t const *b_col = b_tile + row_idx * depth_tile_size;
                        svfloat64_t b_vec = svld1_f64(ptrue_d, b_col);

                        svmopa_za64_f64_m(0, pred_rows, pred_cols, a_vec, b_vec);
                    }
                }

                // Extract partial sums from ZA tile
                for (nk_size_t row_idx = 0; row_idx < tile_dim; row_idx++) {
                    svst1_hor_za64(0, row_idx, ptrue_d, partial + row_idx * tile_dim);
                }

                // Apply Kahan compensation
                for (nk_size_t row_idx = 0; row_idx < tile_dim; row_idx++) {
                    nk_size_t const base_idx = row_idx * tile_dim;

                    svfloat64_t acc_v = svld1_f64(ptrue_d, acc + base_idx);
                    svfloat64_t comp_v = svld1_f64(ptrue_d, comp + base_idx);
                    svfloat64_t part_v = svld1_f64(ptrue_d, partial + base_idx);

                    // Kahan summation: y = part - comp; t = acc + y; comp = (t - acc) - y; acc = t
                    svfloat64_t y = svsub_f64_x(ptrue_d, part_v, comp_v);
                    svfloat64_t t = svadd_f64_x(ptrue_d, acc_v, y);
                    comp_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t, acc_v), y);
                    acc_v = t;

                    svst1_f64(ptrue_d, acc + base_idx, acc_v);
                    svst1_f64(ptrue_d, comp + base_idx, comp_v);
                }
            }

            // Store results with predication
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements + col_start;
                svfloat64_t acc_vec = svld1_f64(ptrue_d, acc + row_idx * tile_dim);
                svst1_f64(pred_cols, c_row, acc_vec);
            }
        }
    }
}

/**
 *  @brief Computes C = A × B^T using packed `f64` B with Kahan compensated summation.
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

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SMEF64_H
